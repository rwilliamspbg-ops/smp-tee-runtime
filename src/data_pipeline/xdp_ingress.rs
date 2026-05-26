use crate::tee_interface::{TeeError, TeeGuard};

#[derive(Debug, Clone, Copy)]
pub struct PacketView<'a> {
    pub data: &'a [u8],
}

#[derive(Debug, Default)]
pub struct XdpIngress;

impl XdpIngress {
    pub fn parse_ring_buffer<'a>(ring_bytes: &'a [u8], frame_size: usize) -> Vec<PacketView<'a>> {
        if frame_size == 0 {
            return Vec::new();
        }

        ring_bytes
            .chunks(frame_size)
            .filter(|frame| !frame.is_empty())
            .map(|frame| PacketView { data: frame })
            .collect()
    }

    pub fn write_packet_into_tee<T: TeeGuard>(
        &self,
        tee: &mut T,
        packet: PacketView<'_>,
    ) -> Result<*mut u8, TeeError> {
        let ptr = tee.allocate_memory(packet.data.len())?;
        tee.write_data(ptr, packet.data)?;
        Ok(ptr)
    }
}

#[cfg(test)]
mod tests {
    use crate::tee_interface::InMemoryTee;

    use super::*;

    #[test]
    fn ring_buffer_parsing_keeps_packet_views() {
        let ring = [1_u8, 2, 3, 4, 5, 6];
        let packets = XdpIngress::parse_ring_buffer(&ring, 2);

        assert_eq!(packets.len(), 3);
        assert_eq!(packets[0].data, &[1, 2]);
        assert_eq!(packets[2].data, &[5, 6]);
    }

    #[test]
    fn ingress_writes_payload_to_tee() {
        let mut tee = InMemoryTee::default();
        tee.initialize().unwrap();

        let ingress = XdpIngress;
        let ptr = ingress
            .write_packet_into_tee(
                &mut tee,
                PacketView {
                    data: &[9, 8, 7, 6],
                },
            )
            .unwrap();

        let output = tee
            .execute_computation(
                &[ptr.cast_const()],
                &crate::tee_interface::ComputationParams {
                    algorithm: crate::tee_interface::AggregationAlgorithm::FederatedAveraging,
                },
            )
            .unwrap();

        assert_eq!(output, vec![9, 8, 7, 6]);
    }
}
