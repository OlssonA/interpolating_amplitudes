module     p2_gg_httbar_d67h4l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d67h4l1d.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd67h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(57) :: acd67
      complex(ki) :: brack
      acd67(1)=dotproduct(k2,qshift)
      acd67(2)=abb67(49)
      acd67(3)=dotproduct(qshift,qshift)
      acd67(4)=abb67(34)
      acd67(5)=abb67(12)
      acd67(6)=dotproduct(qshift,spvak1k2)
      acd67(7)=abb67(23)
      acd67(8)=dotproduct(qshift,spvak1l4)
      acd67(9)=abb67(30)
      acd67(10)=dotproduct(qshift,spvak2k1)
      acd67(11)=abb67(13)
      acd67(12)=dotproduct(qshift,spval5k1)
      acd67(13)=abb67(44)
      acd67(14)=abb67(10)
      acd67(15)=abb67(48)
      acd67(16)=abb67(19)
      acd67(17)=abb67(28)
      acd67(18)=abb67(21)
      acd67(19)=abb67(25)
      acd67(20)=abb67(24)
      acd67(21)=dotproduct(qshift,spvak2l3)
      acd67(22)=abb67(11)
      acd67(23)=dotproduct(qshift,spvak2l4)
      acd67(24)=dotproduct(qshift,spval5k2)
      acd67(25)=abb67(16)
      acd67(26)=abb67(35)
      acd67(27)=dotproduct(qshift,spvak2l5)
      acd67(28)=abb67(17)
      acd67(29)=dotproduct(qshift,spval3k2)
      acd67(30)=abb67(22)
      acd67(31)=dotproduct(qshift,spval3l4)
      acd67(32)=abb67(20)
      acd67(33)=dotproduct(qshift,spval3l5)
      acd67(34)=abb67(26)
      acd67(35)=dotproduct(qshift,spval4k2)
      acd67(36)=abb67(18)
      acd67(37)=dotproduct(qshift,spval4l3)
      acd67(38)=abb67(14)
      acd67(39)=dotproduct(qshift,spval5l3)
      acd67(40)=abb67(45)
      acd67(41)=abb67(15)
      acd67(42)=-acd67(12)*acd67(13)
      acd67(43)=-acd67(10)*acd67(11)
      acd67(44)=-acd67(8)*acd67(9)
      acd67(45)=-acd67(6)*acd67(7)
      acd67(46)=-acd67(1)*acd67(4)
      acd67(42)=acd67(46)+acd67(45)+acd67(44)+acd67(43)+acd67(14)+acd67(42)
      acd67(42)=acd67(3)*acd67(42)
      acd67(43)=acd67(17)*acd67(24)
      acd67(43)=acd67(43)-acd67(25)
      acd67(43)=acd67(23)*acd67(43)
      acd67(44)=-acd67(39)*acd67(40)
      acd67(45)=-acd67(37)*acd67(38)
      acd67(46)=-acd67(35)*acd67(36)
      acd67(47)=-acd67(33)*acd67(34)
      acd67(48)=-acd67(31)*acd67(32)
      acd67(49)=-acd67(29)*acd67(30)
      acd67(50)=-acd67(27)*acd67(28)
      acd67(51)=-acd67(21)*acd67(22)
      acd67(52)=-acd67(24)*acd67(26)
      acd67(53)=-acd67(12)*acd67(20)
      acd67(54)=-acd67(10)*acd67(19)
      acd67(55)=-acd67(12)*acd67(17)
      acd67(55)=-acd67(18)+acd67(55)
      acd67(55)=acd67(8)*acd67(55)
      acd67(56)=acd67(10)*acd67(15)
      acd67(56)=-acd67(16)+acd67(56)
      acd67(56)=acd67(6)*acd67(56)
      acd67(57)=acd67(1)*acd67(2)
      acd67(57)=-acd67(5)+acd67(57)
      acd67(57)=acd67(1)*acd67(57)
      brack=acd67(41)+acd67(42)+acd67(43)+acd67(44)+acd67(45)+acd67(46)+acd67(4&
      &7)+acd67(48)+acd67(49)+acd67(50)+acd67(51)+acd67(52)+acd67(53)+acd67(54)&
      &+acd67(55)+acd67(56)+acd67(57)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd67h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(66) :: acd67
      complex(ki) :: brack
      acd67(1)=k2(iv1)
      acd67(2)=dotproduct(k2,qshift)
      acd67(3)=abb67(49)
      acd67(4)=dotproduct(qshift,qshift)
      acd67(5)=abb67(34)
      acd67(6)=abb67(12)
      acd67(7)=qshift(iv1)
      acd67(8)=dotproduct(qshift,spvak1k2)
      acd67(9)=abb67(23)
      acd67(10)=dotproduct(qshift,spvak1l4)
      acd67(11)=abb67(30)
      acd67(12)=dotproduct(qshift,spvak2k1)
      acd67(13)=abb67(13)
      acd67(14)=dotproduct(qshift,spval5k1)
      acd67(15)=abb67(44)
      acd67(16)=abb67(10)
      acd67(17)=spvak1k2(iv1)
      acd67(18)=abb67(48)
      acd67(19)=abb67(19)
      acd67(20)=spvak1l4(iv1)
      acd67(21)=abb67(28)
      acd67(22)=abb67(21)
      acd67(23)=spvak2k1(iv1)
      acd67(24)=abb67(25)
      acd67(25)=spval5k1(iv1)
      acd67(26)=abb67(24)
      acd67(27)=spvak2l3(iv1)
      acd67(28)=abb67(11)
      acd67(29)=spvak2l4(iv1)
      acd67(30)=dotproduct(qshift,spval5k2)
      acd67(31)=abb67(16)
      acd67(32)=spval5k2(iv1)
      acd67(33)=dotproduct(qshift,spvak2l4)
      acd67(34)=abb67(35)
      acd67(35)=spvak2l5(iv1)
      acd67(36)=abb67(17)
      acd67(37)=spval3k2(iv1)
      acd67(38)=abb67(22)
      acd67(39)=spval3l4(iv1)
      acd67(40)=abb67(20)
      acd67(41)=spval3l5(iv1)
      acd67(42)=abb67(26)
      acd67(43)=spval4k2(iv1)
      acd67(44)=abb67(18)
      acd67(45)=spval4l3(iv1)
      acd67(46)=abb67(14)
      acd67(47)=spval5l3(iv1)
      acd67(48)=abb67(45)
      acd67(49)=acd67(25)*acd67(15)
      acd67(50)=acd67(23)*acd67(13)
      acd67(51)=acd67(20)*acd67(11)
      acd67(52)=acd67(17)*acd67(9)
      acd67(53)=acd67(1)*acd67(5)
      acd67(49)=acd67(53)+acd67(52)+acd67(51)+acd67(49)+acd67(50)
      acd67(49)=acd67(4)*acd67(49)
      acd67(50)=acd67(14)*acd67(15)
      acd67(51)=acd67(12)*acd67(13)
      acd67(52)=acd67(10)*acd67(11)
      acd67(53)=acd67(8)*acd67(9)
      acd67(54)=acd67(2)*acd67(5)
      acd67(50)=acd67(54)+acd67(53)+acd67(52)+acd67(51)-acd67(16)+acd67(50)
      acd67(50)=acd67(7)*acd67(50)
      acd67(51)=-acd67(32)*acd67(33)
      acd67(52)=-acd67(29)*acd67(30)
      acd67(53)=acd67(25)*acd67(10)
      acd67(54)=acd67(20)*acd67(14)
      acd67(51)=acd67(54)+acd67(53)+acd67(51)+acd67(52)
      acd67(51)=acd67(21)*acd67(51)
      acd67(52)=acd67(47)*acd67(48)
      acd67(53)=acd67(45)*acd67(46)
      acd67(54)=acd67(43)*acd67(44)
      acd67(55)=acd67(41)*acd67(42)
      acd67(56)=acd67(39)*acd67(40)
      acd67(57)=acd67(37)*acd67(38)
      acd67(58)=acd67(35)*acd67(36)
      acd67(59)=acd67(27)*acd67(28)
      acd67(60)=acd67(32)*acd67(34)
      acd67(61)=acd67(29)*acd67(31)
      acd67(62)=acd67(25)*acd67(26)
      acd67(63)=-acd67(8)*acd67(18)
      acd67(63)=acd67(24)+acd67(63)
      acd67(63)=acd67(23)*acd67(63)
      acd67(64)=acd67(20)*acd67(22)
      acd67(65)=-acd67(12)*acd67(18)
      acd67(65)=acd67(19)+acd67(65)
      acd67(65)=acd67(17)*acd67(65)
      acd67(66)=acd67(2)*acd67(3)
      acd67(66)=acd67(6)-2.0_ki*acd67(66)
      acd67(66)=acd67(1)*acd67(66)
      brack=acd67(49)+2.0_ki*acd67(50)+acd67(51)+acd67(52)+acd67(53)+acd67(54)+&
      &acd67(55)+acd67(56)+acd67(57)+acd67(58)+acd67(59)+acd67(60)+acd67(61)+ac&
      &d67(62)+acd67(63)+acd67(64)+acd67(65)+acd67(66)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd67h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(38) :: acd67
      complex(ki) :: brack
      acd67(1)=d(iv1,iv2)
      acd67(2)=dotproduct(k2,qshift)
      acd67(3)=abb67(34)
      acd67(4)=dotproduct(qshift,spvak1k2)
      acd67(5)=abb67(23)
      acd67(6)=dotproduct(qshift,spvak1l4)
      acd67(7)=abb67(30)
      acd67(8)=dotproduct(qshift,spvak2k1)
      acd67(9)=abb67(13)
      acd67(10)=dotproduct(qshift,spval5k1)
      acd67(11)=abb67(44)
      acd67(12)=abb67(10)
      acd67(13)=k2(iv1)
      acd67(14)=k2(iv2)
      acd67(15)=abb67(49)
      acd67(16)=qshift(iv2)
      acd67(17)=qshift(iv1)
      acd67(18)=spvak1k2(iv2)
      acd67(19)=spvak1l4(iv2)
      acd67(20)=spvak2k1(iv2)
      acd67(21)=spval5k1(iv2)
      acd67(22)=spvak1k2(iv1)
      acd67(23)=spvak1l4(iv1)
      acd67(24)=spvak2k1(iv1)
      acd67(25)=spval5k1(iv1)
      acd67(26)=abb67(48)
      acd67(27)=abb67(28)
      acd67(28)=spvak2l4(iv1)
      acd67(29)=spval5k2(iv2)
      acd67(30)=spvak2l4(iv2)
      acd67(31)=spval5k2(iv1)
      acd67(32)=-acd67(11)*acd67(21)
      acd67(33)=-acd67(9)*acd67(20)
      acd67(34)=-acd67(7)*acd67(19)
      acd67(35)=-acd67(5)*acd67(18)
      acd67(36)=-acd67(3)*acd67(14)
      acd67(32)=acd67(36)+acd67(35)+acd67(34)+acd67(32)+acd67(33)
      acd67(32)=acd67(17)*acd67(32)
      acd67(33)=-acd67(11)*acd67(25)
      acd67(34)=-acd67(9)*acd67(24)
      acd67(35)=-acd67(7)*acd67(23)
      acd67(36)=-acd67(5)*acd67(22)
      acd67(37)=-acd67(3)*acd67(13)
      acd67(33)=acd67(37)+acd67(36)+acd67(35)+acd67(33)+acd67(34)
      acd67(33)=acd67(16)*acd67(33)
      acd67(34)=-acd67(11)*acd67(10)
      acd67(35)=-acd67(9)*acd67(8)
      acd67(36)=-acd67(7)*acd67(6)
      acd67(37)=-acd67(5)*acd67(4)
      acd67(38)=-acd67(3)*acd67(2)
      acd67(34)=acd67(38)+acd67(37)+acd67(36)+acd67(35)+acd67(12)+acd67(34)
      acd67(34)=acd67(1)*acd67(34)
      acd67(35)=acd67(13)*acd67(14)*acd67(15)
      acd67(32)=acd67(35)+acd67(32)+acd67(33)+acd67(34)
      acd67(33)=acd67(30)*acd67(31)
      acd67(34)=acd67(28)*acd67(29)
      acd67(35)=-acd67(21)*acd67(23)
      acd67(36)=-acd67(19)*acd67(25)
      acd67(33)=acd67(36)+acd67(35)+acd67(33)+acd67(34)
      acd67(33)=acd67(27)*acd67(33)
      acd67(34)=acd67(20)*acd67(22)
      acd67(35)=acd67(18)*acd67(24)
      acd67(34)=acd67(34)+acd67(35)
      acd67(34)=acd67(26)*acd67(34)
      brack=2.0_ki*acd67(32)+acd67(33)+acd67(34)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd67h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(30) :: acd67
      complex(ki) :: brack
      acd67(1)=d(iv1,iv2)
      acd67(2)=k2(iv3)
      acd67(3)=abb67(34)
      acd67(4)=spvak1k2(iv3)
      acd67(5)=abb67(23)
      acd67(6)=spvak1l4(iv3)
      acd67(7)=abb67(30)
      acd67(8)=spvak2k1(iv3)
      acd67(9)=abb67(13)
      acd67(10)=spval5k1(iv3)
      acd67(11)=abb67(44)
      acd67(12)=d(iv1,iv3)
      acd67(13)=k2(iv2)
      acd67(14)=spvak1k2(iv2)
      acd67(15)=spvak1l4(iv2)
      acd67(16)=spvak2k1(iv2)
      acd67(17)=spval5k1(iv2)
      acd67(18)=d(iv2,iv3)
      acd67(19)=k2(iv1)
      acd67(20)=spvak1k2(iv1)
      acd67(21)=spvak1l4(iv1)
      acd67(22)=spvak2k1(iv1)
      acd67(23)=spval5k1(iv1)
      acd67(24)=acd67(2)*acd67(3)
      acd67(25)=acd67(4)*acd67(5)
      acd67(26)=acd67(6)*acd67(7)
      acd67(27)=acd67(8)*acd67(9)
      acd67(28)=acd67(10)*acd67(11)
      acd67(24)=acd67(28)+acd67(27)+acd67(26)+acd67(24)+acd67(25)
      acd67(24)=acd67(1)*acd67(24)
      acd67(25)=acd67(13)*acd67(3)
      acd67(26)=acd67(14)*acd67(5)
      acd67(27)=acd67(15)*acd67(7)
      acd67(28)=acd67(16)*acd67(9)
      acd67(29)=acd67(17)*acd67(11)
      acd67(25)=acd67(29)+acd67(28)+acd67(27)+acd67(26)+acd67(25)
      acd67(25)=acd67(12)*acd67(25)
      acd67(26)=acd67(19)*acd67(3)
      acd67(27)=acd67(20)*acd67(5)
      acd67(28)=acd67(21)*acd67(7)
      acd67(29)=acd67(22)*acd67(9)
      acd67(30)=acd67(23)*acd67(11)
      acd67(26)=acd67(30)+acd67(29)+acd67(28)+acd67(27)+acd67(26)
      acd67(26)=acd67(18)*acd67(26)
      acd67(24)=acd67(26)+acd67(25)+acd67(24)
      brack=2.0_ki*acd67(24)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd67h4
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
      qshift = k4
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
end module     p2_gg_httbar_d67h4l1d
