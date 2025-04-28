module     p2_gg_httbar_d2h0l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d2h0l1d.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd2h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(19) :: acd2
      complex(ki) :: brack
      acd2(1)=dotproduct(qshift,spval3k2)
      acd2(2)=abb2(8)
      acd2(3)=dotproduct(qshift,spval4k2)
      acd2(4)=abb2(7)
      acd2(5)=dotproduct(qshift,spval4l3)
      acd2(6)=abb2(13)
      acd2(7)=dotproduct(qshift,spval5k2)
      acd2(8)=abb2(11)
      acd2(9)=dotproduct(qshift,spvae1e2)
      acd2(10)=abb2(10)
      acd2(11)=dotproduct(qshift,spvae2e1)
      acd2(12)=abb2(21)
      acd2(13)=abb2(12)
      acd2(14)=-acd2(2)*acd2(1)
      acd2(15)=-acd2(4)*acd2(3)
      acd2(16)=-acd2(6)*acd2(5)
      acd2(17)=-acd2(8)*acd2(7)
      acd2(18)=-acd2(10)*acd2(9)
      acd2(19)=-acd2(12)*acd2(11)
      brack=acd2(13)+acd2(14)+acd2(15)+acd2(16)+acd2(17)+acd2(18)+acd2(19)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd2h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(18) :: acd2
      complex(ki) :: brack
      acd2(1)=spval3k2(iv1)
      acd2(2)=abb2(8)
      acd2(3)=spval4k2(iv1)
      acd2(4)=abb2(7)
      acd2(5)=spval4l3(iv1)
      acd2(6)=abb2(13)
      acd2(7)=spval5k2(iv1)
      acd2(8)=abb2(11)
      acd2(9)=spvae1e2(iv1)
      acd2(10)=abb2(10)
      acd2(11)=spvae2e1(iv1)
      acd2(12)=abb2(21)
      acd2(13)=-acd2(2)*acd2(1)
      acd2(14)=-acd2(4)*acd2(3)
      acd2(15)=-acd2(6)*acd2(5)
      acd2(16)=-acd2(8)*acd2(7)
      acd2(17)=-acd2(10)*acd2(9)
      acd2(18)=-acd2(12)*acd2(11)
      brack=acd2(13)+acd2(14)+acd2(15)+acd2(16)+acd2(17)+acd2(18)
   end function brack_2
!---#] function brack_2:
!---#[ function derivative:
   function derivative(mu2,i1) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd2h0
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = -k3-k5
      numerator = 0.0_ki
      deg = 0
      if(present(i1)) then
          iv1=i1
          deg=1
      else
          iv1=1
      end if
      t1 = 0
      if(deg.eq.0) then
         numerator = cond(epspow.eq.t1,brack_1,Q,mu2)
         return
      end if
      if(deg.eq.1) then
         numerator = cond(epspow.eq.t1,brack_2,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p2_gg_httbar_d2h0l1d
