module     p0_ubaru_httbar_d71h1l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity1d71h1l1d.f90
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
      use p0_ubaru_httbar_abbrevd71h1
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(11) :: acd71
      complex(ki) :: brack
      acd71(1)=dotproduct(l3,qshift)
      acd71(2)=abb71(10)
      acd71(3)=dotproduct(qshift,qshift)
      acd71(4)=dotproduct(qshift,spval3k2)
      acd71(5)=abb71(8)
      acd71(6)=dotproduct(qshift,spval5k2)
      acd71(7)=abb71(7)
      acd71(8)=abb71(15)
      acd71(9)=acd71(1)-acd71(3)
      acd71(9)=acd71(2)*acd71(9)
      acd71(10)=-acd71(5)*acd71(4)
      acd71(11)=-acd71(7)*acd71(6)
      brack=acd71(8)+acd71(9)+acd71(10)+acd71(11)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd71h1
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(10) :: acd71
      complex(ki) :: brack
      acd71(1)=l3(iv1)
      acd71(2)=abb71(10)
      acd71(3)=qshift(iv1)
      acd71(4)=spval3k2(iv1)
      acd71(5)=abb71(8)
      acd71(6)=spval5k2(iv1)
      acd71(7)=abb71(7)
      acd71(8)=acd71(1)-2.0_ki*acd71(3)
      acd71(8)=acd71(2)*acd71(8)
      acd71(9)=-acd71(5)*acd71(4)
      acd71(10)=-acd71(7)*acd71(6)
      brack=acd71(8)+acd71(9)+acd71(10)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd71h1
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(3) :: acd71
      complex(ki) :: brack
      acd71(1)=d(iv1,iv2)
      acd71(2)=abb71(10)
      brack=-2.0_ki*acd71(2)*acd71(1)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd71h1
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
      qshift = -k5
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
end module     p0_ubaru_httbar_d71h1l1d
