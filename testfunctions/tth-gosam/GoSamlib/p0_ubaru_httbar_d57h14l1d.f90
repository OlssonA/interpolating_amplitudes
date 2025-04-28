module     p0_ubaru_httbar_d57h14l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity14d57h14l1d.f90
   ! generator: buildfortran_d.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   integer, private :: iv3
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd57h14
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(42) :: acd57
      complex(ki) :: brack
      acd57(1)=dotproduct(k2,qshift)
      acd57(2)=dotproduct(qshift,spvak2l5)
      acd57(3)=abb57(24)
      acd57(4)=abb57(36)
      acd57(5)=dotproduct(l3,qshift)
      acd57(6)=abb57(29)
      acd57(7)=dotproduct(qshift,qshift)
      acd57(8)=abb57(20)
      acd57(9)=dotproduct(qshift,spvak2k1)
      acd57(10)=abb57(15)
      acd57(11)=dotproduct(qshift,spvak2l4)
      acd57(12)=abb57(26)
      acd57(13)=abb57(12)
      acd57(14)=abb57(18)
      acd57(15)=abb57(17)
      acd57(16)=abb57(10)
      acd57(17)=dotproduct(qshift,spval3l4)
      acd57(18)=abb57(19)
      acd57(19)=dotproduct(qshift,spval3l5)
      acd57(20)=abb57(27)
      acd57(21)=abb57(11)
      acd57(22)=abb57(13)
      acd57(23)=dotproduct(qshift,spvak1l3)
      acd57(24)=abb57(16)
      acd57(25)=abb57(23)
      acd57(26)=dotproduct(qshift,spvak2l3)
      acd57(27)=abb57(25)
      acd57(28)=abb57(9)
      acd57(29)=dotproduct(qshift,spval5l3)
      acd57(30)=abb57(30)
      acd57(31)=abb57(14)
      acd57(32)=-acd57(10)*acd57(7)
      acd57(33)=acd57(14)*acd57(2)
      acd57(34)=acd57(16)*acd57(11)
      acd57(35)=acd57(18)*acd57(17)
      acd57(36)=acd57(20)*acd57(19)
      acd57(32)=-acd57(21)+acd57(36)+acd57(35)+acd57(34)+acd57(33)+acd57(32)
      acd57(32)=acd57(9)*acd57(32)
      acd57(33)=acd57(8)*acd57(2)
      acd57(34)=-acd57(12)*acd57(11)
      acd57(33)=acd57(13)+acd57(34)+acd57(33)
      acd57(33)=acd57(7)*acd57(33)
      acd57(34)=acd57(3)*acd57(2)
      acd57(34)=-acd57(4)+acd57(34)
      acd57(34)=acd57(1)*acd57(34)
      acd57(35)=-acd57(8)*acd57(19)
      acd57(35)=-acd57(28)+acd57(35)
      acd57(35)=acd57(26)*acd57(35)
      acd57(36)=-acd57(6)*acd57(5)
      acd57(37)=-acd57(15)*acd57(2)
      acd57(38)=-acd57(22)*acd57(11)
      acd57(39)=-acd57(24)*acd57(23)
      acd57(40)=-acd57(25)*acd57(17)
      acd57(41)=-acd57(27)*acd57(19)
      acd57(42)=-acd57(30)*acd57(29)
      brack=acd57(31)+acd57(32)+acd57(33)+acd57(34)+acd57(35)+acd57(36)+acd57(3&
      &7)+acd57(38)+acd57(39)+acd57(40)+acd57(41)+acd57(42)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd57h14
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(51) :: acd57
      complex(ki) :: brack
      acd57(1)=k2(iv1)
      acd57(2)=dotproduct(qshift,spvak2l5)
      acd57(3)=abb57(24)
      acd57(4)=abb57(36)
      acd57(5)=l3(iv1)
      acd57(6)=abb57(29)
      acd57(7)=qshift(iv1)
      acd57(8)=abb57(20)
      acd57(9)=dotproduct(qshift,spvak2k1)
      acd57(10)=abb57(15)
      acd57(11)=dotproduct(qshift,spvak2l4)
      acd57(12)=abb57(26)
      acd57(13)=abb57(12)
      acd57(14)=spvak2l5(iv1)
      acd57(15)=dotproduct(k2,qshift)
      acd57(16)=dotproduct(qshift,qshift)
      acd57(17)=abb57(18)
      acd57(18)=abb57(17)
      acd57(19)=spvak2k1(iv1)
      acd57(20)=abb57(10)
      acd57(21)=dotproduct(qshift,spval3l4)
      acd57(22)=abb57(19)
      acd57(23)=dotproduct(qshift,spval3l5)
      acd57(24)=abb57(27)
      acd57(25)=abb57(11)
      acd57(26)=spvak2l4(iv1)
      acd57(27)=abb57(13)
      acd57(28)=spvak1l3(iv1)
      acd57(29)=abb57(16)
      acd57(30)=spval3l4(iv1)
      acd57(31)=abb57(23)
      acd57(32)=spval3l5(iv1)
      acd57(33)=dotproduct(qshift,spvak2l3)
      acd57(34)=abb57(25)
      acd57(35)=spvak2l3(iv1)
      acd57(36)=abb57(9)
      acd57(37)=spval5l3(iv1)
      acd57(38)=abb57(30)
      acd57(39)=-acd57(22)*acd57(30)
      acd57(40)=-acd57(32)*acd57(24)
      acd57(41)=-acd57(26)*acd57(20)
      acd57(42)=-acd57(14)*acd57(17)
      acd57(43)=2.0_ki*acd57(7)
      acd57(44)=acd57(10)*acd57(43)
      acd57(39)=acd57(44)+acd57(42)+acd57(41)+acd57(39)+acd57(40)
      acd57(39)=acd57(9)*acd57(39)
      acd57(40)=-acd57(23)*acd57(24)
      acd57(41)=-acd57(22)*acd57(21)
      acd57(42)=-acd57(11)*acd57(20)
      acd57(44)=acd57(16)*acd57(10)
      acd57(45)=-acd57(2)*acd57(17)
      acd57(40)=acd57(45)+acd57(44)+acd57(42)+acd57(41)+acd57(25)+acd57(40)
      acd57(40)=acd57(19)*acd57(40)
      acd57(41)=acd57(23)*acd57(35)
      acd57(42)=acd57(32)*acd57(33)
      acd57(44)=-acd57(14)*acd57(16)
      acd57(41)=acd57(44)+acd57(41)+acd57(42)
      acd57(41)=acd57(8)*acd57(41)
      acd57(42)=acd57(11)*acd57(12)
      acd57(44)=-acd57(8)*acd57(2)
      acd57(42)=acd57(44)-acd57(13)+acd57(42)
      acd57(42)=acd57(42)*acd57(43)
      acd57(43)=-acd57(2)*acd57(3)
      acd57(43)=acd57(43)+acd57(4)
      acd57(43)=acd57(1)*acd57(43)
      acd57(44)=acd57(16)*acd57(12)
      acd57(44)=acd57(44)+acd57(27)
      acd57(44)=acd57(26)*acd57(44)
      acd57(45)=acd57(37)*acd57(38)
      acd57(46)=acd57(28)*acd57(29)
      acd57(47)=acd57(5)*acd57(6)
      acd57(48)=acd57(35)*acd57(36)
      acd57(49)=acd57(30)*acd57(31)
      acd57(50)=acd57(32)*acd57(34)
      acd57(51)=-acd57(3)*acd57(15)
      acd57(51)=acd57(18)+acd57(51)
      acd57(51)=acd57(14)*acd57(51)
      brack=acd57(39)+acd57(40)+acd57(41)+acd57(42)+acd57(43)+acd57(44)+acd57(4&
      &5)+acd57(46)+acd57(47)+acd57(48)+acd57(49)+acd57(50)+acd57(51)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd57h14
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(38) :: acd57
      complex(ki) :: brack
      acd57(1)=d(iv1,iv2)
      acd57(2)=dotproduct(qshift,spvak2k1)
      acd57(3)=abb57(15)
      acd57(4)=dotproduct(qshift,spvak2l4)
      acd57(5)=abb57(26)
      acd57(6)=dotproduct(qshift,spvak2l5)
      acd57(7)=abb57(20)
      acd57(8)=abb57(12)
      acd57(9)=k2(iv1)
      acd57(10)=spvak2l5(iv2)
      acd57(11)=abb57(24)
      acd57(12)=k2(iv2)
      acd57(13)=spvak2l5(iv1)
      acd57(14)=qshift(iv1)
      acd57(15)=spvak2k1(iv2)
      acd57(16)=spvak2l4(iv2)
      acd57(17)=qshift(iv2)
      acd57(18)=spvak2k1(iv1)
      acd57(19)=spvak2l4(iv1)
      acd57(20)=abb57(10)
      acd57(21)=abb57(18)
      acd57(22)=spval3l4(iv2)
      acd57(23)=abb57(19)
      acd57(24)=spval3l5(iv2)
      acd57(25)=abb57(27)
      acd57(26)=spval3l4(iv1)
      acd57(27)=spval3l5(iv1)
      acd57(28)=spvak2l3(iv2)
      acd57(29)=spvak2l3(iv1)
      acd57(30)=-acd57(3)*acd57(15)
      acd57(31)=acd57(10)*acd57(7)
      acd57(32)=-acd57(16)*acd57(5)
      acd57(30)=acd57(32)+acd57(30)+acd57(31)
      acd57(30)=acd57(14)*acd57(30)
      acd57(31)=-acd57(3)*acd57(18)
      acd57(32)=acd57(13)*acd57(7)
      acd57(33)=-acd57(19)*acd57(5)
      acd57(31)=acd57(33)+acd57(31)+acd57(32)
      acd57(31)=acd57(17)*acd57(31)
      acd57(30)=acd57(31)+acd57(30)
      acd57(31)=-acd57(2)*acd57(3)
      acd57(32)=-acd57(4)*acd57(5)
      acd57(33)=acd57(6)*acd57(7)
      acd57(31)=acd57(8)+acd57(33)+acd57(32)+acd57(31)
      acd57(32)=2.0_ki*acd57(1)
      acd57(31)=acd57(32)*acd57(31)
      acd57(32)=acd57(9)*acd57(10)
      acd57(33)=acd57(12)*acd57(13)
      acd57(32)=acd57(33)+acd57(32)
      acd57(32)=acd57(11)*acd57(32)
      acd57(33)=acd57(22)*acd57(18)
      acd57(34)=acd57(26)*acd57(15)
      acd57(33)=acd57(34)+acd57(33)
      acd57(33)=acd57(23)*acd57(33)
      acd57(34)=acd57(25)*acd57(18)
      acd57(35)=-acd57(29)*acd57(7)
      acd57(34)=acd57(35)+acd57(34)
      acd57(34)=acd57(24)*acd57(34)
      acd57(35)=acd57(25)*acd57(15)
      acd57(36)=-acd57(28)*acd57(7)
      acd57(35)=acd57(36)+acd57(35)
      acd57(35)=acd57(27)*acd57(35)
      acd57(36)=acd57(16)*acd57(18)
      acd57(37)=acd57(19)*acd57(15)
      acd57(36)=acd57(36)+acd57(37)
      acd57(36)=acd57(20)*acd57(36)
      acd57(37)=acd57(10)*acd57(18)
      acd57(38)=acd57(13)*acd57(15)
      acd57(37)=acd57(37)+acd57(38)
      acd57(37)=acd57(21)*acd57(37)
      brack=2.0_ki*acd57(30)+acd57(31)+acd57(32)+acd57(33)+acd57(34)+acd57(35)+&
      &acd57(36)+acd57(37)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd57h14
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(20) :: acd57
      complex(ki) :: brack
      acd57(1)=d(iv1,iv2)
      acd57(2)=spvak2k1(iv3)
      acd57(3)=abb57(15)
      acd57(4)=spvak2l4(iv3)
      acd57(5)=abb57(26)
      acd57(6)=spvak2l5(iv3)
      acd57(7)=abb57(20)
      acd57(8)=d(iv1,iv3)
      acd57(9)=spvak2k1(iv2)
      acd57(10)=spvak2l4(iv2)
      acd57(11)=spvak2l5(iv2)
      acd57(12)=d(iv2,iv3)
      acd57(13)=spvak2k1(iv1)
      acd57(14)=spvak2l4(iv1)
      acd57(15)=spvak2l5(iv1)
      acd57(16)=acd57(2)*acd57(3)
      acd57(17)=acd57(4)*acd57(5)
      acd57(18)=-acd57(6)*acd57(7)
      acd57(16)=acd57(18)+acd57(16)+acd57(17)
      acd57(16)=acd57(1)*acd57(16)
      acd57(17)=acd57(9)*acd57(3)
      acd57(18)=acd57(10)*acd57(5)
      acd57(19)=-acd57(11)*acd57(7)
      acd57(17)=acd57(19)+acd57(18)+acd57(17)
      acd57(17)=acd57(8)*acd57(17)
      acd57(18)=acd57(13)*acd57(3)
      acd57(19)=acd57(14)*acd57(5)
      acd57(20)=-acd57(15)*acd57(7)
      acd57(18)=acd57(20)+acd57(19)+acd57(18)
      acd57(18)=acd57(12)*acd57(18)
      acd57(16)=acd57(18)+acd57(17)+acd57(16)
      brack=2.0_ki*acd57(16)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd57h14
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      integer, intent(in), optional :: i3
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = k3+k5
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
      if(present(i3)) then
          iv3=i3
          deg=3
      else
          iv3=1
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
      if(deg.eq.3) then
         numerator = cond(epspow.eq.t1,brack_4,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p0_ubaru_httbar_d57h14l1d
