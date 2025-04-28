module     p2_gg_httbar_d13h12l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d13h12l1d.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd13h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(56) :: acd13
      complex(ki) :: brack
      acd13(1)=dotproduct(k1,qshift)
      acd13(2)=dotproduct(qshift,spvak2l3)
      acd13(3)=abb13(47)
      acd13(4)=dotproduct(qshift,spvak2l4)
      acd13(5)=abb13(24)
      acd13(6)=dotproduct(qshift,spvak2l5)
      acd13(7)=abb13(39)
      acd13(8)=dotproduct(qshift,spval3l4)
      acd13(9)=abb13(34)
      acd13(10)=abb13(22)
      acd13(11)=dotproduct(k2,qshift)
      acd13(12)=abb13(13)
      acd13(13)=dotproduct(l4,qshift)
      acd13(14)=abb13(15)
      acd13(15)=dotproduct(qshift,qshift)
      acd13(16)=abb13(23)
      acd13(17)=abb13(20)
      acd13(18)=abb13(16)
      acd13(19)=abb13(19)
      acd13(20)=abb13(37)
      acd13(21)=dotproduct(qshift,spvak1k2)
      acd13(22)=abb13(21)
      acd13(23)=dotproduct(qshift,spvak1l3)
      acd13(24)=abb13(12)
      acd13(25)=dotproduct(qshift,spvak1l4)
      acd13(26)=abb13(11)
      acd13(27)=dotproduct(qshift,spvak1l5)
      acd13(28)=abb13(18)
      acd13(29)=dotproduct(qshift,spvak2k1)
      acd13(30)=abb13(10)
      acd13(31)=dotproduct(qshift,spval3k1)
      acd13(32)=abb13(14)
      acd13(33)=dotproduct(qshift,spval3k2)
      acd13(34)=abb13(17)
      acd13(35)=dotproduct(qshift,spval4l3)
      acd13(36)=abb13(38)
      acd13(37)=dotproduct(qshift,spval4l5)
      acd13(38)=abb13(36)
      acd13(39)=abb13(9)
      acd13(40)=acd13(8)*acd13(9)
      acd13(41)=acd13(6)*acd13(7)
      acd13(42)=acd13(4)*acd13(5)
      acd13(43)=acd13(2)*acd13(3)
      acd13(40)=acd13(40)-acd13(41)+acd13(42)-acd13(43)
      acd13(41)=-acd13(12)-acd13(40)
      acd13(41)=acd13(11)*acd13(41)
      acd13(40)=-acd13(10)+acd13(40)
      acd13(40)=acd13(1)*acd13(40)
      acd13(42)=-acd13(37)*acd13(38)
      acd13(43)=-acd13(35)*acd13(36)
      acd13(44)=-acd13(33)*acd13(34)
      acd13(45)=-acd13(31)*acd13(32)
      acd13(46)=-acd13(29)*acd13(30)
      acd13(47)=-acd13(27)*acd13(28)
      acd13(48)=-acd13(25)*acd13(26)
      acd13(49)=-acd13(23)*acd13(24)
      acd13(50)=-acd13(21)*acd13(22)
      acd13(51)=acd13(15)*acd13(16)
      acd13(52)=-acd13(13)*acd13(14)
      acd13(53)=-acd13(8)*acd13(20)
      acd13(54)=-acd13(6)*acd13(19)
      acd13(55)=-acd13(4)*acd13(18)
      acd13(56)=-acd13(2)*acd13(17)
      brack=acd13(39)+acd13(40)+acd13(41)+acd13(42)+acd13(43)+acd13(44)+acd13(4&
      &5)+acd13(46)+acd13(47)+acd13(48)+acd13(49)+acd13(50)+acd13(51)+acd13(52)&
      &+acd13(53)+acd13(54)+acd13(55)+acd13(56)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd13h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(62) :: acd13
      complex(ki) :: brack
      acd13(1)=k1(iv1)
      acd13(2)=dotproduct(qshift,spvak2l3)
      acd13(3)=abb13(47)
      acd13(4)=dotproduct(qshift,spvak2l4)
      acd13(5)=abb13(24)
      acd13(6)=dotproduct(qshift,spvak2l5)
      acd13(7)=abb13(39)
      acd13(8)=dotproduct(qshift,spval3l4)
      acd13(9)=abb13(34)
      acd13(10)=abb13(22)
      acd13(11)=k2(iv1)
      acd13(12)=abb13(13)
      acd13(13)=l4(iv1)
      acd13(14)=abb13(15)
      acd13(15)=qshift(iv1)
      acd13(16)=abb13(23)
      acd13(17)=spvak2l3(iv1)
      acd13(18)=dotproduct(k1,qshift)
      acd13(19)=dotproduct(k2,qshift)
      acd13(20)=abb13(20)
      acd13(21)=spvak2l4(iv1)
      acd13(22)=abb13(16)
      acd13(23)=spvak2l5(iv1)
      acd13(24)=abb13(19)
      acd13(25)=spval3l4(iv1)
      acd13(26)=abb13(37)
      acd13(27)=spvak1k2(iv1)
      acd13(28)=abb13(21)
      acd13(29)=spvak1l3(iv1)
      acd13(30)=abb13(12)
      acd13(31)=spvak1l4(iv1)
      acd13(32)=abb13(11)
      acd13(33)=spvak1l5(iv1)
      acd13(34)=abb13(18)
      acd13(35)=spvak2k1(iv1)
      acd13(36)=abb13(10)
      acd13(37)=spval3k1(iv1)
      acd13(38)=abb13(14)
      acd13(39)=spval3k2(iv1)
      acd13(40)=abb13(17)
      acd13(41)=spval4l3(iv1)
      acd13(42)=abb13(38)
      acd13(43)=spval4l5(iv1)
      acd13(44)=abb13(36)
      acd13(45)=-acd13(2)*acd13(3)
      acd13(46)=acd13(4)*acd13(5)
      acd13(47)=-acd13(6)*acd13(7)
      acd13(48)=acd13(8)*acd13(9)
      acd13(45)=acd13(48)+acd13(47)+acd13(46)+acd13(45)
      acd13(46)=acd13(11)-acd13(1)
      acd13(45)=acd13(46)*acd13(45)
      acd13(46)=acd13(19)-acd13(18)
      acd13(47)=-acd13(3)*acd13(46)
      acd13(47)=acd13(20)+acd13(47)
      acd13(47)=acd13(17)*acd13(47)
      acd13(48)=acd13(5)*acd13(46)
      acd13(48)=acd13(22)+acd13(48)
      acd13(48)=acd13(21)*acd13(48)
      acd13(49)=-acd13(7)*acd13(46)
      acd13(49)=acd13(24)+acd13(49)
      acd13(49)=acd13(23)*acd13(49)
      acd13(46)=acd13(9)*acd13(46)
      acd13(46)=acd13(26)+acd13(46)
      acd13(46)=acd13(25)*acd13(46)
      acd13(50)=acd13(10)*acd13(1)
      acd13(51)=acd13(12)*acd13(11)
      acd13(52)=acd13(14)*acd13(13)
      acd13(53)=acd13(16)*acd13(15)
      acd13(54)=acd13(28)*acd13(27)
      acd13(55)=acd13(30)*acd13(29)
      acd13(56)=acd13(32)*acd13(31)
      acd13(57)=acd13(34)*acd13(33)
      acd13(58)=acd13(36)*acd13(35)
      acd13(59)=acd13(38)*acd13(37)
      acd13(60)=acd13(40)*acd13(39)
      acd13(61)=acd13(42)*acd13(41)
      acd13(62)=acd13(44)*acd13(43)
      brack=acd13(45)+acd13(46)+acd13(47)+acd13(48)+acd13(49)+acd13(50)+acd13(5&
      &1)+acd13(52)-2.0_ki*acd13(53)+acd13(54)+acd13(55)+acd13(56)+acd13(57)+ac&
      &d13(58)+acd13(59)+acd13(60)+acd13(61)+acd13(62)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd13h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(23) :: acd13
      complex(ki) :: brack
      acd13(1)=d(iv1,iv2)
      acd13(2)=abb13(23)
      acd13(3)=k1(iv1)
      acd13(4)=spvak2l3(iv2)
      acd13(5)=abb13(47)
      acd13(6)=spvak2l4(iv2)
      acd13(7)=abb13(24)
      acd13(8)=spvak2l5(iv2)
      acd13(9)=abb13(39)
      acd13(10)=spval3l4(iv2)
      acd13(11)=abb13(34)
      acd13(12)=k1(iv2)
      acd13(13)=spvak2l3(iv1)
      acd13(14)=spvak2l4(iv1)
      acd13(15)=spvak2l5(iv1)
      acd13(16)=spval3l4(iv1)
      acd13(17)=k2(iv1)
      acd13(18)=k2(iv2)
      acd13(19)=acd13(13)*acd13(5)
      acd13(20)=-acd13(14)*acd13(7)
      acd13(21)=acd13(15)*acd13(9)
      acd13(22)=-acd13(16)*acd13(11)
      acd13(19)=acd13(22)+acd13(21)+acd13(20)+acd13(19)
      acd13(20)=acd13(18)-acd13(12)
      acd13(19)=acd13(20)*acd13(19)
      acd13(20)=acd13(4)*acd13(5)
      acd13(21)=-acd13(6)*acd13(7)
      acd13(22)=acd13(8)*acd13(9)
      acd13(23)=-acd13(10)*acd13(11)
      acd13(20)=acd13(23)+acd13(22)+acd13(20)+acd13(21)
      acd13(21)=acd13(17)-acd13(3)
      acd13(20)=acd13(21)*acd13(20)
      acd13(21)=acd13(2)*acd13(1)
      brack=acd13(19)+acd13(20)+2.0_ki*acd13(21)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd13h12
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
      qshift = k3+k4+k5
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
end module     p2_gg_httbar_d13h12l1d
