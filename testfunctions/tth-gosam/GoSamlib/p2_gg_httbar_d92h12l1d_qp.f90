module     p2_gg_httbar_d92h12l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d92h12l1d_qp.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond, d => metric_tensor
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd92h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd92
      complex(ki) :: brack
      acd92(1)=abb92(13)
      brack=acd92(1)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd92h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(57) :: acd92
      complex(ki) :: brack
      acd92(1)=k2(iv1)
      acd92(2)=abb92(51)
      acd92(3)=l4(iv1)
      acd92(4)=abb92(40)
      acd92(5)=e1(iv1)
      acd92(6)=abb92(21)
      acd92(7)=spvak2l3(iv1)
      acd92(8)=abb92(20)
      acd92(9)=spvak2l4(iv1)
      acd92(10)=abb92(19)
      acd92(11)=spvak2l5(iv1)
      acd92(12)=abb92(64)
      acd92(13)=spval3k2(iv1)
      acd92(14)=abb92(50)
      acd92(15)=spval3l4(iv1)
      acd92(16)=abb92(49)
      acd92(17)=spval4l3(iv1)
      acd92(18)=abb92(42)
      acd92(19)=spval4l5(iv1)
      acd92(20)=abb92(37)
      acd92(21)=spvak1e2(iv1)
      acd92(22)=abb92(30)
      acd92(23)=spvae2k1(iv1)
      acd92(24)=abb92(16)
      acd92(25)=spvak2e2(iv1)
      acd92(26)=abb92(18)
      acd92(27)=spval3e2(iv1)
      acd92(28)=abb92(66)
      acd92(29)=spvae2l3(iv1)
      acd92(30)=abb92(53)
      acd92(31)=spvae2l4(iv1)
      acd92(32)=abb92(27)
      acd92(33)=spvae2l5(iv1)
      acd92(34)=abb92(52)
      acd92(35)=spvae1e2(iv1)
      acd92(36)=abb92(44)
      acd92(37)=spvae2e1(iv1)
      acd92(38)=abb92(28)
      acd92(39)=-acd92(2)*acd92(1)
      acd92(40)=-acd92(4)*acd92(3)
      acd92(41)=-acd92(6)*acd92(5)
      acd92(42)=-acd92(8)*acd92(7)
      acd92(43)=-acd92(10)*acd92(9)
      acd92(44)=-acd92(12)*acd92(11)
      acd92(45)=-acd92(14)*acd92(13)
      acd92(46)=-acd92(16)*acd92(15)
      acd92(47)=-acd92(18)*acd92(17)
      acd92(48)=-acd92(20)*acd92(19)
      acd92(49)=-acd92(22)*acd92(21)
      acd92(50)=-acd92(24)*acd92(23)
      acd92(51)=-acd92(26)*acd92(25)
      acd92(52)=-acd92(28)*acd92(27)
      acd92(53)=-acd92(30)*acd92(29)
      acd92(54)=-acd92(32)*acd92(31)
      acd92(55)=-acd92(34)*acd92(33)
      acd92(56)=-acd92(36)*acd92(35)
      acd92(57)=-acd92(38)*acd92(37)
      brack=acd92(39)+acd92(40)+acd92(41)+acd92(42)+acd92(43)+acd92(44)+acd92(4&
      &5)+acd92(46)+acd92(47)+acd92(48)+acd92(49)+acd92(50)+acd92(51)+acd92(52)&
      &+acd92(53)+acd92(54)+acd92(55)+acd92(56)+acd92(57)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd92h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(73) :: acd92
      complex(ki) :: brack
      acd92(1)=d(iv1,iv2)
      acd92(2)=abb92(22)
      acd92(3)=k2(iv1)
      acd92(4)=spvae1e2(iv2)
      acd92(5)=abb92(25)
      acd92(6)=k2(iv2)
      acd92(7)=spvae1e2(iv1)
      acd92(8)=l4(iv1)
      acd92(9)=spvae2e1(iv2)
      acd92(10)=abb92(48)
      acd92(11)=l4(iv2)
      acd92(12)=spvae2e1(iv1)
      acd92(13)=e1(iv1)
      acd92(14)=spvak2e2(iv2)
      acd92(15)=abb92(11)
      acd92(16)=spval3e2(iv2)
      acd92(17)=abb92(10)
      acd92(18)=spvae2l3(iv2)
      acd92(19)=abb92(62)
      acd92(20)=spvae2l4(iv2)
      acd92(21)=abb92(17)
      acd92(22)=spvae2l5(iv2)
      acd92(23)=abb92(9)
      acd92(24)=e1(iv2)
      acd92(25)=spvak2e2(iv1)
      acd92(26)=spval3e2(iv1)
      acd92(27)=spvae2l3(iv1)
      acd92(28)=spvae2l4(iv1)
      acd92(29)=spvae2l5(iv1)
      acd92(30)=abb92(63)
      acd92(31)=abb92(36)
      acd92(32)=abb92(58)
      acd92(33)=spvak2l4(iv2)
      acd92(34)=abb92(14)
      acd92(35)=spval3k2(iv2)
      acd92(36)=abb92(57)
      acd92(37)=spval3l4(iv2)
      acd92(38)=abb92(56)
      acd92(39)=spvae2k1(iv2)
      acd92(40)=abb92(15)
      acd92(41)=spvak2l4(iv1)
      acd92(42)=spval3k2(iv1)
      acd92(43)=spval3l4(iv1)
      acd92(44)=spvae2k1(iv1)
      acd92(45)=abb92(24)
      acd92(46)=abb92(70)
      acd92(47)=spvak2l3(iv2)
      acd92(48)=abb92(26)
      acd92(49)=abb92(23)
      acd92(50)=spvak2l5(iv2)
      acd92(51)=abb92(60)
      acd92(52)=spval4l3(iv2)
      acd92(53)=abb92(46)
      acd92(54)=spval4l5(iv2)
      acd92(55)=abb92(39)
      acd92(56)=spvak1e2(iv2)
      acd92(57)=abb92(34)
      acd92(58)=spvak2l3(iv1)
      acd92(59)=spvak2l5(iv1)
      acd92(60)=spval4l3(iv1)
      acd92(61)=spval4l5(iv1)
      acd92(62)=spvak1e2(iv1)
      acd92(63)=acd92(57)*acd92(56)
      acd92(64)=acd92(55)*acd92(54)
      acd92(65)=acd92(53)*acd92(52)
      acd92(66)=acd92(51)*acd92(50)
      acd92(67)=acd92(48)*acd92(47)
      acd92(68)=acd92(33)*acd92(49)
      acd92(69)=acd92(16)*acd92(46)
      acd92(70)=acd92(14)*acd92(45)
      acd92(71)=-acd92(10)*acd92(11)
      acd92(63)=acd92(71)+acd92(70)+acd92(69)+acd92(68)+acd92(67)+acd92(66)+acd&
      &92(65)+acd92(63)+acd92(64)
      acd92(63)=acd92(12)*acd92(63)
      acd92(64)=acd92(57)*acd92(62)
      acd92(65)=acd92(55)*acd92(61)
      acd92(66)=acd92(53)*acd92(60)
      acd92(67)=acd92(51)*acd92(59)
      acd92(68)=acd92(48)*acd92(58)
      acd92(69)=acd92(41)*acd92(49)
      acd92(70)=acd92(26)*acd92(46)
      acd92(71)=acd92(25)*acd92(45)
      acd92(72)=-acd92(10)*acd92(8)
      acd92(64)=acd92(72)+acd92(71)+acd92(70)+acd92(69)+acd92(68)+acd92(67)+acd&
      &92(66)+acd92(64)+acd92(65)
      acd92(64)=acd92(9)*acd92(64)
      acd92(65)=acd92(40)*acd92(39)
      acd92(66)=acd92(38)*acd92(37)
      acd92(67)=acd92(36)*acd92(35)
      acd92(68)=acd92(33)*acd92(34)
      acd92(69)=acd92(22)*acd92(32)
      acd92(70)=acd92(20)*acd92(31)
      acd92(71)=acd92(18)*acd92(30)
      acd92(72)=acd92(5)*acd92(6)
      acd92(65)=acd92(72)+acd92(71)+acd92(70)+acd92(69)+acd92(68)+acd92(67)+acd&
      &92(65)+acd92(66)
      acd92(65)=acd92(7)*acd92(65)
      acd92(66)=acd92(40)*acd92(44)
      acd92(67)=acd92(38)*acd92(43)
      acd92(68)=acd92(36)*acd92(42)
      acd92(69)=acd92(34)*acd92(41)
      acd92(70)=acd92(29)*acd92(32)
      acd92(71)=acd92(28)*acd92(31)
      acd92(72)=acd92(27)*acd92(30)
      acd92(73)=acd92(5)*acd92(3)
      acd92(66)=acd92(73)+acd92(72)+acd92(71)+acd92(70)+acd92(69)+acd92(68)+acd&
      &92(66)+acd92(67)
      acd92(66)=acd92(4)*acd92(66)
      acd92(67)=acd92(23)*acd92(29)
      acd92(68)=acd92(21)*acd92(28)
      acd92(69)=acd92(19)*acd92(27)
      acd92(70)=acd92(17)*acd92(26)
      acd92(71)=acd92(15)*acd92(25)
      acd92(67)=acd92(71)+acd92(70)+acd92(69)+acd92(67)+acd92(68)
      acd92(67)=acd92(24)*acd92(67)
      acd92(68)=acd92(22)*acd92(23)
      acd92(69)=acd92(20)*acd92(21)
      acd92(70)=acd92(18)*acd92(19)
      acd92(71)=acd92(16)*acd92(17)
      acd92(72)=acd92(14)*acd92(15)
      acd92(68)=acd92(72)+acd92(71)+acd92(70)+acd92(68)+acd92(69)
      acd92(68)=acd92(13)*acd92(68)
      acd92(69)=acd92(1)*acd92(2)
      brack=acd92(63)+acd92(64)+acd92(65)+acd92(66)+acd92(67)+acd92(68)+2.0_ki*&
      &acd92(69)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd92h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(43) :: acd92
      complex(ki) :: brack
      acd92(1)=d(iv1,iv2)
      acd92(2)=spvae1e2(iv3)
      acd92(3)=abb92(45)
      acd92(4)=spvae2e1(iv3)
      acd92(5)=abb92(43)
      acd92(6)=d(iv1,iv3)
      acd92(7)=spvae1e2(iv2)
      acd92(8)=spvae2e1(iv2)
      acd92(9)=d(iv2,iv3)
      acd92(10)=spvae1e2(iv1)
      acd92(11)=spvae2e1(iv1)
      acd92(12)=e1(iv1)
      acd92(13)=spvak2e2(iv2)
      acd92(14)=spvae2l3(iv3)
      acd92(15)=abb92(12)
      acd92(16)=spvae2l4(iv3)
      acd92(17)=abb92(8)
      acd92(18)=spvae2l5(iv3)
      acd92(19)=abb92(31)
      acd92(20)=spvak2e2(iv3)
      acd92(21)=spvae2l3(iv2)
      acd92(22)=spvae2l4(iv2)
      acd92(23)=spvae2l5(iv2)
      acd92(24)=spval3e2(iv3)
      acd92(25)=abb92(73)
      acd92(26)=spval3e2(iv2)
      acd92(27)=e1(iv2)
      acd92(28)=spvak2e2(iv1)
      acd92(29)=spvae2l3(iv1)
      acd92(30)=spvae2l4(iv1)
      acd92(31)=spvae2l5(iv1)
      acd92(32)=spval3e2(iv1)
      acd92(33)=e1(iv3)
      acd92(34)=-acd92(16)*acd92(17)
      acd92(35)=-acd92(14)*acd92(15)
      acd92(36)=-acd92(18)*acd92(19)
      acd92(34)=acd92(36)+acd92(35)+acd92(34)
      acd92(35)=acd92(13)*acd92(12)
      acd92(36)=acd92(28)*acd92(27)
      acd92(35)=acd92(35)+acd92(36)
      acd92(34)=acd92(35)*acd92(34)
      acd92(35)=-acd92(30)*acd92(17)
      acd92(36)=-acd92(29)*acd92(15)
      acd92(37)=-acd92(31)*acd92(19)
      acd92(35)=acd92(37)+acd92(36)+acd92(35)
      acd92(36)=acd92(13)*acd92(33)
      acd92(37)=acd92(20)*acd92(27)
      acd92(36)=acd92(36)+acd92(37)
      acd92(35)=acd92(36)*acd92(35)
      acd92(36)=-acd92(22)*acd92(17)
      acd92(37)=-acd92(21)*acd92(15)
      acd92(38)=-acd92(23)*acd92(19)
      acd92(36)=acd92(38)+acd92(37)+acd92(36)
      acd92(37)=acd92(20)*acd92(12)
      acd92(38)=acd92(28)*acd92(33)
      acd92(37)=acd92(37)+acd92(38)
      acd92(36)=acd92(37)*acd92(36)
      acd92(37)=-acd92(4)*acd92(1)
      acd92(38)=-acd92(8)*acd92(6)
      acd92(39)=-acd92(11)*acd92(9)
      acd92(37)=acd92(39)+acd92(38)+acd92(37)
      acd92(38)=2.0_ki*acd92(5)
      acd92(37)=acd92(38)*acd92(37)
      acd92(38)=-acd92(2)*acd92(1)
      acd92(39)=-acd92(7)*acd92(6)
      acd92(40)=-acd92(10)*acd92(9)
      acd92(38)=acd92(40)+acd92(39)+acd92(38)
      acd92(39)=2.0_ki*acd92(3)
      acd92(38)=acd92(39)*acd92(38)
      acd92(39)=acd92(22)*acd92(25)
      acd92(40)=-acd92(12)*acd92(39)
      acd92(41)=acd92(30)*acd92(25)
      acd92(42)=-acd92(27)*acd92(41)
      acd92(40)=acd92(40)+acd92(42)
      acd92(40)=acd92(24)*acd92(40)
      acd92(42)=acd92(16)*acd92(25)
      acd92(43)=-acd92(12)*acd92(42)
      acd92(41)=-acd92(33)*acd92(41)
      acd92(41)=acd92(43)+acd92(41)
      acd92(41)=acd92(26)*acd92(41)
      acd92(42)=-acd92(27)*acd92(42)
      acd92(39)=-acd92(33)*acd92(39)
      acd92(39)=acd92(42)+acd92(39)
      acd92(39)=acd92(32)*acd92(39)
      brack=acd92(34)+acd92(35)+acd92(36)+acd92(37)+acd92(38)+acd92(39)+acd92(4&
      &0)+acd92(41)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd92h12_qp
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
end module     p2_gg_httbar_d92h12l1d_qp
