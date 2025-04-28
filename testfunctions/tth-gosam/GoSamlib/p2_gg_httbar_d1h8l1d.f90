module     p2_gg_httbar_d1h8l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d1h8l1d.f90
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
      use p2_gg_httbar_abbrevd1h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(19) :: acd1
      complex(ki) :: brack
      acd1(1)=dotproduct(k2,qshift)
      acd1(2)=abb1(21)
      acd1(3)=dotproduct(qshift,spvak2l3)
      acd1(4)=abb1(11)
      acd1(5)=dotproduct(qshift,spval3l5)
      acd1(6)=abb1(9)
      acd1(7)=dotproduct(qshift,spval4l5)
      acd1(8)=abb1(8)
      acd1(9)=dotproduct(qshift,spvae1e2)
      acd1(10)=abb1(7)
      acd1(11)=dotproduct(qshift,spvae2e1)
      acd1(12)=abb1(13)
      acd1(13)=abb1(10)
      acd1(14)=-acd1(2)*acd1(1)
      acd1(15)=-acd1(4)*acd1(3)
      acd1(16)=-acd1(6)*acd1(5)
      acd1(17)=-acd1(8)*acd1(7)
      acd1(18)=-acd1(10)*acd1(9)
      acd1(19)=-acd1(12)*acd1(11)
      brack=acd1(13)+acd1(14)+acd1(15)+acd1(16)+acd1(17)+acd1(18)+acd1(19)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd1h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(18) :: acd1
      complex(ki) :: brack
      acd1(1)=k2(iv1)
      acd1(2)=abb1(21)
      acd1(3)=spvak2l3(iv1)
      acd1(4)=abb1(11)
      acd1(5)=spval3l5(iv1)
      acd1(6)=abb1(9)
      acd1(7)=spval4l5(iv1)
      acd1(8)=abb1(8)
      acd1(9)=spvae1e2(iv1)
      acd1(10)=abb1(7)
      acd1(11)=spvae2e1(iv1)
      acd1(12)=abb1(13)
      acd1(13)=acd1(2)*acd1(1)
      acd1(14)=acd1(4)*acd1(3)
      acd1(15)=acd1(6)*acd1(5)
      acd1(16)=acd1(8)*acd1(7)
      acd1(17)=acd1(10)*acd1(9)
      acd1(18)=acd1(12)*acd1(11)
      brack=acd1(13)+acd1(14)+acd1(15)+acd1(16)+acd1(17)+acd1(18)
   end function brack_2
!---#] function brack_2:
!---#[ function derivative:
   function derivative(mu2,i1) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd1h8
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = k5
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
end module     p2_gg_httbar_d1h8l1d
