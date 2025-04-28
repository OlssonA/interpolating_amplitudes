module     p2_gg_httbar_globalsh8_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_color_qp, only:&
      & c1v => c1,&
      & c2v => c2,&
      & c3v => c3

   implicit none
   private
   complex(ki), public :: c1
   complex(ki), public :: c2
   complex(ki), public :: c3

   public :: init_lo

   complex(ki), public :: rat2
contains

subroutine     init_lo()
   use p2_gg_httbar_globalsl1_qp, only: epspow, ccontract, amp0
   implicit none
   c1 = ccontract(c1v, amp0)
   c2 = ccontract(c2v, amp0)
   c3 = ccontract(c3v, amp0)
end subroutine init_lo

end module p2_gg_httbar_globalsh8_qp
