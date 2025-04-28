module     p0_ubaru_httbar_d22h10l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity10d22h10l1d.f90
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
      use p0_ubaru_httbar_abbrevd22h10
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd22
      complex(ki) :: brack
      acd22(1)=abb22(17)
      brack=acd22(1)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd22h10
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(27) :: acd22
      complex(ki) :: brack
      acd22(1)=k2(iv1)
      acd22(2)=abb22(13)
      acd22(3)=l3(iv1)
      acd22(4)=abb22(21)
      acd22(5)=l5(iv1)
      acd22(6)=abb22(18)
      acd22(7)=spvak2k1(iv1)
      acd22(8)=abb22(20)
      acd22(9)=spvak2l3(iv1)
      acd22(10)=abb22(14)
      acd22(11)=spvak2l5(iv1)
      acd22(12)=abb22(12)
      acd22(13)=spval3k1(iv1)
      acd22(14)=abb22(32)
      acd22(15)=spval3l5(iv1)
      acd22(16)=abb22(16)
      acd22(17)=spval5l3(iv1)
      acd22(18)=abb22(15)
      acd22(19)=-acd22(2)*acd22(1)
      acd22(20)=-acd22(4)*acd22(3)
      acd22(21)=-acd22(6)*acd22(5)
      acd22(22)=-acd22(8)*acd22(7)
      acd22(23)=-acd22(10)*acd22(9)
      acd22(24)=-acd22(12)*acd22(11)
      acd22(25)=-acd22(14)*acd22(13)
      acd22(26)=-acd22(16)*acd22(15)
      acd22(27)=-acd22(18)*acd22(17)
      brack=acd22(19)+acd22(20)+acd22(21)+acd22(22)+acd22(23)+acd22(24)+acd22(2&
      &5)+acd22(26)+acd22(27)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd22h10
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd22
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd22h10
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
      qshift = 0
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
end module     p0_ubaru_httbar_d22h10l1d
