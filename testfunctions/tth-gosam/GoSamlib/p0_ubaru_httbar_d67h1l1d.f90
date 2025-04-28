module     p0_ubaru_httbar_d67h1l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity1d67h1l1d.f90
   ! generator: buildfortran_d.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd67h1
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(9) :: acd67
      complex(ki) :: brack
      acd67(1)=dotproduct(qshift,spvak1k2)
      acd67(2)=dotproduct(qshift,spval4k2)
      acd67(3)=abb67(8)
      acd67(4)=dotproduct(qshift,spval4l3)
      acd67(5)=abb67(9)
      acd67(6)=abb67(7)
      acd67(7)=abb67(15)
      acd67(8)=acd67(3)*acd67(1)
      acd67(8)=-acd67(6)+acd67(8)
      acd67(8)=acd67(2)*acd67(8)
      acd67(9)=acd67(5)*acd67(1)
      acd67(9)=-acd67(7)+acd67(9)
      acd67(9)=acd67(4)*acd67(9)
      brack=acd67(8)+acd67(9)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd67h1
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(13) :: acd67
      complex(ki) :: brack
      acd67(1)=spvak1k2(iv1)
      acd67(2)=dotproduct(qshift,spval4k2)
      acd67(3)=abb67(8)
      acd67(4)=dotproduct(qshift,spval4l3)
      acd67(5)=abb67(9)
      acd67(6)=spval4k2(iv1)
      acd67(7)=dotproduct(qshift,spvak1k2)
      acd67(8)=abb67(7)
      acd67(9)=spval4l3(iv1)
      acd67(10)=abb67(15)
      acd67(11)=acd67(2)*acd67(3)
      acd67(12)=acd67(4)*acd67(5)
      acd67(11)=acd67(12)+acd67(11)
      acd67(11)=acd67(1)*acd67(11)
      acd67(12)=acd67(7)*acd67(3)
      acd67(12)=-acd67(8)+acd67(12)
      acd67(12)=acd67(6)*acd67(12)
      acd67(13)=acd67(7)*acd67(5)
      acd67(13)=-acd67(10)+acd67(13)
      acd67(13)=acd67(9)*acd67(13)
      brack=acd67(11)+acd67(12)+acd67(13)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd67h1
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(11) :: acd67
      complex(ki) :: brack
      acd67(1)=spvak1k2(iv1)
      acd67(2)=spval4k2(iv2)
      acd67(3)=abb67(8)
      acd67(4)=spval4l3(iv2)
      acd67(5)=abb67(9)
      acd67(6)=spvak1k2(iv2)
      acd67(7)=spval4k2(iv1)
      acd67(8)=spval4l3(iv1)
      acd67(9)=acd67(2)*acd67(3)
      acd67(10)=acd67(4)*acd67(5)
      acd67(9)=acd67(9)+acd67(10)
      acd67(9)=acd67(1)*acd67(9)
      acd67(10)=acd67(7)*acd67(3)
      acd67(11)=acd67(8)*acd67(5)
      acd67(10)=acd67(11)+acd67(10)
      acd67(10)=acd67(6)*acd67(10)
      brack=acd67(9)+acd67(10)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd67h1
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = -k2
      numerator = 0.0_ki
      deg = 0
      if(present(i1)) then
          iv1=i1
          deg=1
      else
          iv1=1
      end if
      if(present(i2)) then
          iv2=i2
          deg=2
      else
          iv2=1
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
      if(deg.eq.2) then
         numerator = cond(epspow.eq.t1,brack_3,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p0_ubaru_httbar_d67h1l1d
