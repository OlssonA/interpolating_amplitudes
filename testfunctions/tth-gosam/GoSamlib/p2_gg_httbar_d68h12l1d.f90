module     p2_gg_httbar_d68h12l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d68h12l1d.f90
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
      use p2_gg_httbar_abbrevd68h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(48) :: acd68
      complex(ki) :: brack
      acd68(1)=dotproduct(k2,qshift)
      acd68(2)=dotproduct(qshift,spvak2l4)
      acd68(3)=abb68(25)
      acd68(4)=dotproduct(qshift,spvak2l5)
      acd68(5)=abb68(47)
      acd68(6)=abb68(13)
      acd68(7)=dotproduct(l4,qshift)
      acd68(8)=abb68(16)
      acd68(9)=dotproduct(l5,qshift)
      acd68(10)=abb68(20)
      acd68(11)=dotproduct(qshift,qshift)
      acd68(12)=abb68(28)
      acd68(13)=abb68(44)
      acd68(14)=dotproduct(qshift,spvak1l4)
      acd68(15)=abb68(12)
      acd68(16)=dotproduct(qshift,spvak1l5)
      acd68(17)=abb68(21)
      acd68(18)=abb68(10)
      acd68(19)=abb68(17)
      acd68(20)=abb68(29)
      acd68(21)=dotproduct(qshift,spvak2k1)
      acd68(22)=abb68(22)
      acd68(23)=abb68(18)
      acd68(24)=abb68(46)
      acd68(25)=abb68(19)
      acd68(26)=abb68(11)
      acd68(27)=dotproduct(qshift,spvak2l3)
      acd68(28)=abb68(32)
      acd68(29)=dotproduct(qshift,spval3l4)
      acd68(30)=abb68(24)
      acd68(31)=dotproduct(qshift,spval4l5)
      acd68(32)=abb68(14)
      acd68(33)=dotproduct(qshift,spval5l4)
      acd68(34)=abb68(15)
      acd68(35)=abb68(23)
      acd68(36)=-acd68(12)*acd68(2)
      acd68(37)=-acd68(13)*acd68(4)
      acd68(38)=-acd68(15)*acd68(14)
      acd68(39)=-acd68(17)*acd68(16)
      acd68(36)=acd68(18)+acd68(39)+acd68(38)+acd68(37)+acd68(36)
      acd68(36)=acd68(11)*acd68(36)
      acd68(37)=acd68(3)*acd68(2)
      acd68(38)=acd68(5)*acd68(4)
      acd68(37)=-acd68(6)+acd68(37)+acd68(38)
      acd68(37)=acd68(1)*acd68(37)
      acd68(38)=-acd68(22)*acd68(14)
      acd68(39)=acd68(24)*acd68(16)
      acd68(38)=-acd68(26)+acd68(39)+acd68(38)
      acd68(38)=acd68(21)*acd68(38)
      acd68(39)=-acd68(8)*acd68(7)
      acd68(40)=-acd68(10)*acd68(9)
      acd68(41)=-acd68(19)*acd68(2)
      acd68(42)=-acd68(20)*acd68(4)
      acd68(43)=-acd68(23)*acd68(14)
      acd68(44)=-acd68(25)*acd68(16)
      acd68(45)=-acd68(28)*acd68(27)
      acd68(46)=-acd68(30)*acd68(29)
      acd68(47)=-acd68(32)*acd68(31)
      acd68(48)=-acd68(34)*acd68(33)
      brack=acd68(35)+acd68(36)+acd68(37)+acd68(38)+acd68(39)+acd68(40)+acd68(4&
      &1)+acd68(42)+acd68(43)+acd68(44)+acd68(45)+acd68(46)+acd68(47)+acd68(48)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd68h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(55) :: acd68
      complex(ki) :: brack
      acd68(1)=k2(iv1)
      acd68(2)=dotproduct(qshift,spvak2l4)
      acd68(3)=abb68(25)
      acd68(4)=dotproduct(qshift,spvak2l5)
      acd68(5)=abb68(47)
      acd68(6)=abb68(13)
      acd68(7)=l4(iv1)
      acd68(8)=abb68(16)
      acd68(9)=l5(iv1)
      acd68(10)=abb68(20)
      acd68(11)=qshift(iv1)
      acd68(12)=abb68(28)
      acd68(13)=abb68(44)
      acd68(14)=dotproduct(qshift,spvak1l4)
      acd68(15)=abb68(12)
      acd68(16)=dotproduct(qshift,spvak1l5)
      acd68(17)=abb68(21)
      acd68(18)=abb68(10)
      acd68(19)=spvak2l4(iv1)
      acd68(20)=dotproduct(k2,qshift)
      acd68(21)=dotproduct(qshift,qshift)
      acd68(22)=abb68(17)
      acd68(23)=spvak2l5(iv1)
      acd68(24)=abb68(29)
      acd68(25)=spvak1l4(iv1)
      acd68(26)=dotproduct(qshift,spvak2k1)
      acd68(27)=abb68(22)
      acd68(28)=abb68(18)
      acd68(29)=spvak1l5(iv1)
      acd68(30)=abb68(46)
      acd68(31)=abb68(19)
      acd68(32)=spvak2k1(iv1)
      acd68(33)=abb68(11)
      acd68(34)=spvak2l3(iv1)
      acd68(35)=abb68(32)
      acd68(36)=spval3l4(iv1)
      acd68(37)=abb68(24)
      acd68(38)=spval4l5(iv1)
      acd68(39)=abb68(14)
      acd68(40)=spval5l4(iv1)
      acd68(41)=abb68(15)
      acd68(42)=-acd68(29)*acd68(17)
      acd68(43)=-acd68(25)*acd68(15)
      acd68(44)=-acd68(23)*acd68(13)
      acd68(45)=-acd68(19)*acd68(12)
      acd68(42)=acd68(45)+acd68(44)+acd68(42)+acd68(43)
      acd68(42)=acd68(21)*acd68(42)
      acd68(43)=-acd68(16)*acd68(17)
      acd68(44)=-acd68(14)*acd68(15)
      acd68(45)=-acd68(4)*acd68(13)
      acd68(46)=-acd68(2)*acd68(12)
      acd68(43)=acd68(46)+acd68(45)+acd68(44)+acd68(18)+acd68(43)
      acd68(43)=acd68(11)*acd68(43)
      acd68(44)=acd68(16)*acd68(30)
      acd68(45)=-acd68(14)*acd68(27)
      acd68(44)=acd68(45)-acd68(33)+acd68(44)
      acd68(44)=acd68(32)*acd68(44)
      acd68(45)=acd68(4)*acd68(5)
      acd68(46)=acd68(2)*acd68(3)
      acd68(45)=acd68(46)-acd68(6)+acd68(45)
      acd68(45)=acd68(1)*acd68(45)
      acd68(46)=-acd68(40)*acd68(41)
      acd68(47)=-acd68(38)*acd68(39)
      acd68(48)=-acd68(36)*acd68(37)
      acd68(49)=-acd68(34)*acd68(35)
      acd68(50)=-acd68(9)*acd68(10)
      acd68(51)=-acd68(7)*acd68(8)
      acd68(52)=acd68(26)*acd68(30)
      acd68(52)=-acd68(31)+acd68(52)
      acd68(52)=acd68(29)*acd68(52)
      acd68(53)=-acd68(26)*acd68(27)
      acd68(53)=-acd68(28)+acd68(53)
      acd68(53)=acd68(25)*acd68(53)
      acd68(54)=acd68(5)*acd68(20)
      acd68(54)=-acd68(24)+acd68(54)
      acd68(54)=acd68(23)*acd68(54)
      acd68(55)=acd68(3)*acd68(20)
      acd68(55)=-acd68(22)+acd68(55)
      acd68(55)=acd68(19)*acd68(55)
      brack=acd68(42)+2.0_ki*acd68(43)+acd68(44)+acd68(45)+acd68(46)+acd68(47)+&
      &acd68(48)+acd68(49)+acd68(50)+acd68(51)+acd68(52)+acd68(53)+acd68(54)+ac&
      &d68(55)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd68h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(34) :: acd68
      complex(ki) :: brack
      acd68(1)=d(iv1,iv2)
      acd68(2)=dotproduct(qshift,spvak1l4)
      acd68(3)=abb68(12)
      acd68(4)=dotproduct(qshift,spvak1l5)
      acd68(5)=abb68(21)
      acd68(6)=dotproduct(qshift,spvak2l4)
      acd68(7)=abb68(28)
      acd68(8)=dotproduct(qshift,spvak2l5)
      acd68(9)=abb68(44)
      acd68(10)=abb68(10)
      acd68(11)=k2(iv1)
      acd68(12)=spvak2l4(iv2)
      acd68(13)=abb68(25)
      acd68(14)=spvak2l5(iv2)
      acd68(15)=abb68(47)
      acd68(16)=k2(iv2)
      acd68(17)=spvak2l4(iv1)
      acd68(18)=spvak2l5(iv1)
      acd68(19)=qshift(iv1)
      acd68(20)=spvak1l4(iv2)
      acd68(21)=spvak1l5(iv2)
      acd68(22)=qshift(iv2)
      acd68(23)=spvak1l4(iv1)
      acd68(24)=spvak1l5(iv1)
      acd68(25)=spvak2k1(iv2)
      acd68(26)=abb68(22)
      acd68(27)=spvak2k1(iv1)
      acd68(28)=abb68(46)
      acd68(29)=-acd68(9)*acd68(18)
      acd68(30)=-acd68(7)*acd68(17)
      acd68(31)=-acd68(5)*acd68(24)
      acd68(32)=-acd68(3)*acd68(23)
      acd68(29)=acd68(32)+acd68(31)+acd68(29)+acd68(30)
      acd68(29)=acd68(22)*acd68(29)
      acd68(30)=-acd68(9)*acd68(14)
      acd68(31)=-acd68(7)*acd68(12)
      acd68(32)=-acd68(5)*acd68(21)
      acd68(33)=-acd68(3)*acd68(20)
      acd68(30)=acd68(33)+acd68(32)+acd68(30)+acd68(31)
      acd68(30)=acd68(19)*acd68(30)
      acd68(31)=-acd68(9)*acd68(8)
      acd68(32)=-acd68(7)*acd68(6)
      acd68(33)=-acd68(5)*acd68(4)
      acd68(34)=-acd68(3)*acd68(2)
      acd68(31)=acd68(34)+acd68(33)+acd68(32)+acd68(10)+acd68(31)
      acd68(31)=acd68(1)*acd68(31)
      acd68(29)=acd68(31)+acd68(29)+acd68(30)
      acd68(30)=acd68(21)*acd68(28)
      acd68(31)=-acd68(20)*acd68(26)
      acd68(30)=acd68(31)+acd68(30)
      acd68(30)=acd68(27)*acd68(30)
      acd68(31)=acd68(24)*acd68(28)
      acd68(32)=-acd68(23)*acd68(26)
      acd68(31)=acd68(31)+acd68(32)
      acd68(31)=acd68(25)*acd68(31)
      acd68(32)=acd68(15)*acd68(18)
      acd68(33)=acd68(13)*acd68(17)
      acd68(32)=acd68(33)+acd68(32)
      acd68(32)=acd68(16)*acd68(32)
      acd68(33)=acd68(14)*acd68(15)
      acd68(34)=acd68(12)*acd68(13)
      acd68(33)=acd68(33)+acd68(34)
      acd68(33)=acd68(11)*acd68(33)
      brack=2.0_ki*acd68(29)+acd68(30)+acd68(31)+acd68(32)+acd68(33)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd68h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(25) :: acd68
      complex(ki) :: brack
      acd68(1)=d(iv1,iv2)
      acd68(2)=spvak1l4(iv3)
      acd68(3)=abb68(12)
      acd68(4)=spvak1l5(iv3)
      acd68(5)=abb68(21)
      acd68(6)=spvak2l4(iv3)
      acd68(7)=abb68(28)
      acd68(8)=spvak2l5(iv3)
      acd68(9)=abb68(44)
      acd68(10)=d(iv1,iv3)
      acd68(11)=spvak1l4(iv2)
      acd68(12)=spvak1l5(iv2)
      acd68(13)=spvak2l4(iv2)
      acd68(14)=spvak2l5(iv2)
      acd68(15)=d(iv2,iv3)
      acd68(16)=spvak1l4(iv1)
      acd68(17)=spvak1l5(iv1)
      acd68(18)=spvak2l4(iv1)
      acd68(19)=spvak2l5(iv1)
      acd68(20)=-acd68(2)*acd68(3)
      acd68(21)=-acd68(4)*acd68(5)
      acd68(22)=-acd68(6)*acd68(7)
      acd68(23)=-acd68(8)*acd68(9)
      acd68(20)=acd68(23)+acd68(22)+acd68(20)+acd68(21)
      acd68(20)=acd68(1)*acd68(20)
      acd68(21)=-acd68(11)*acd68(3)
      acd68(22)=-acd68(12)*acd68(5)
      acd68(23)=-acd68(13)*acd68(7)
      acd68(24)=-acd68(14)*acd68(9)
      acd68(21)=acd68(24)+acd68(23)+acd68(22)+acd68(21)
      acd68(21)=acd68(10)*acd68(21)
      acd68(22)=-acd68(16)*acd68(3)
      acd68(23)=-acd68(17)*acd68(5)
      acd68(24)=-acd68(18)*acd68(7)
      acd68(25)=-acd68(19)*acd68(9)
      acd68(22)=acd68(25)+acd68(24)+acd68(23)+acd68(22)
      acd68(22)=acd68(15)*acd68(22)
      acd68(20)=acd68(22)+acd68(21)+acd68(20)
      brack=2.0_ki*acd68(20)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd68h12
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
      qshift = k5
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
end module     p2_gg_httbar_d68h12l1d
