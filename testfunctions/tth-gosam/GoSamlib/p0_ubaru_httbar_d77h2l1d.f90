module     p0_ubaru_httbar_d77h2l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity2d77h2l1d.f90
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
      use p0_ubaru_httbar_abbrevd77h2
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(37) :: acd77
      complex(ki) :: brack
      acd77(1)=dotproduct(k2,qshift)
      acd77(2)=dotproduct(qshift,spval3k1)
      acd77(3)=abb77(38)
      acd77(4)=dotproduct(qshift,spval4k1)
      acd77(5)=abb77(40)
      acd77(6)=dotproduct(qshift,spval5k1)
      acd77(7)=abb77(26)
      acd77(8)=abb77(15)
      acd77(9)=dotproduct(qshift,qshift)
      acd77(10)=abb77(11)
      acd77(11)=abb77(36)
      acd77(12)=dotproduct(qshift,spvak2l3)
      acd77(13)=abb77(16)
      acd77(14)=abb77(30)
      acd77(15)=abb77(44)
      acd77(16)=abb77(22)
      acd77(17)=dotproduct(qshift,spvak2k1)
      acd77(18)=dotproduct(qshift,spval3k2)
      acd77(19)=abb77(12)
      acd77(20)=dotproduct(qshift,spval4k2)
      acd77(21)=abb77(21)
      acd77(22)=dotproduct(qshift,spval4l3)
      acd77(23)=abb77(14)
      acd77(24)=dotproduct(qshift,spval5k2)
      acd77(25)=abb77(20)
      acd77(26)=dotproduct(qshift,spval5l3)
      acd77(27)=abb77(13)
      acd77(28)=abb77(10)
      acd77(29)=abb77(17)
      acd77(30)=abb77(19)
      acd77(31)=acd77(19)*acd77(18)
      acd77(32)=acd77(21)*acd77(20)
      acd77(33)=acd77(23)*acd77(22)
      acd77(34)=acd77(25)*acd77(24)
      acd77(35)=acd77(27)*acd77(26)
      acd77(31)=-acd77(28)+acd77(35)+acd77(34)+acd77(33)+acd77(32)+acd77(31)
      acd77(31)=acd77(17)*acd77(31)
      acd77(32)=acd77(3)*acd77(2)
      acd77(33)=acd77(5)*acd77(4)
      acd77(34)=acd77(7)*acd77(6)
      acd77(32)=-acd77(8)+acd77(34)+acd77(32)+acd77(33)
      acd77(32)=acd77(1)*acd77(32)
      acd77(33)=acd77(13)*acd77(4)
      acd77(34)=acd77(15)*acd77(6)
      acd77(33)=-acd77(29)+acd77(34)+acd77(33)
      acd77(33)=acd77(12)*acd77(33)
      acd77(34)=acd77(10)*acd77(9)
      acd77(35)=-acd77(11)*acd77(2)
      acd77(36)=-acd77(14)*acd77(4)
      acd77(37)=-acd77(16)*acd77(6)
      brack=acd77(30)+acd77(31)+acd77(32)+acd77(33)+acd77(34)+acd77(35)+acd77(3&
      &6)+acd77(37)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd77h2
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(49) :: acd77
      complex(ki) :: brack
      acd77(1)=k2(iv1)
      acd77(2)=dotproduct(qshift,spval3k1)
      acd77(3)=abb77(38)
      acd77(4)=dotproduct(qshift,spval4k1)
      acd77(5)=abb77(40)
      acd77(6)=dotproduct(qshift,spval5k1)
      acd77(7)=abb77(26)
      acd77(8)=abb77(15)
      acd77(9)=qshift(iv1)
      acd77(10)=abb77(11)
      acd77(11)=spval3k1(iv1)
      acd77(12)=dotproduct(k2,qshift)
      acd77(13)=abb77(36)
      acd77(14)=spval4k1(iv1)
      acd77(15)=dotproduct(qshift,spvak2l3)
      acd77(16)=abb77(16)
      acd77(17)=abb77(30)
      acd77(18)=spval5k1(iv1)
      acd77(19)=abb77(44)
      acd77(20)=abb77(22)
      acd77(21)=spvak2k1(iv1)
      acd77(22)=dotproduct(qshift,spval3k2)
      acd77(23)=abb77(12)
      acd77(24)=dotproduct(qshift,spval4k2)
      acd77(25)=abb77(21)
      acd77(26)=dotproduct(qshift,spval4l3)
      acd77(27)=abb77(14)
      acd77(28)=dotproduct(qshift,spval5k2)
      acd77(29)=abb77(20)
      acd77(30)=dotproduct(qshift,spval5l3)
      acd77(31)=abb77(13)
      acd77(32)=abb77(10)
      acd77(33)=spval3k2(iv1)
      acd77(34)=dotproduct(qshift,spvak2k1)
      acd77(35)=spval4k2(iv1)
      acd77(36)=spval4l3(iv1)
      acd77(37)=spval5k2(iv1)
      acd77(38)=spval5l3(iv1)
      acd77(39)=spvak2l3(iv1)
      acd77(40)=abb77(17)
      acd77(41)=-acd77(31)*acd77(38)
      acd77(42)=-acd77(29)*acd77(37)
      acd77(43)=-acd77(27)*acd77(36)
      acd77(44)=-acd77(25)*acd77(35)
      acd77(45)=-acd77(23)*acd77(33)
      acd77(41)=acd77(45)+acd77(44)+acd77(43)+acd77(41)+acd77(42)
      acd77(41)=acd77(34)*acd77(41)
      acd77(42)=-acd77(31)*acd77(30)
      acd77(43)=-acd77(29)*acd77(28)
      acd77(44)=-acd77(27)*acd77(26)
      acd77(45)=-acd77(25)*acd77(24)
      acd77(46)=-acd77(23)*acd77(22)
      acd77(42)=acd77(46)+acd77(45)+acd77(44)+acd77(43)+acd77(32)+acd77(42)
      acd77(42)=acd77(21)*acd77(42)
      acd77(43)=-acd77(3)*acd77(11)
      acd77(44)=-acd77(18)*acd77(7)
      acd77(45)=-acd77(14)*acd77(5)
      acd77(43)=acd77(45)+acd77(43)+acd77(44)
      acd77(43)=acd77(12)*acd77(43)
      acd77(44)=-acd77(6)*acd77(7)
      acd77(45)=-acd77(4)*acd77(5)
      acd77(46)=-acd77(3)*acd77(2)
      acd77(44)=acd77(46)+acd77(45)+acd77(8)+acd77(44)
      acd77(44)=acd77(1)*acd77(44)
      acd77(45)=-acd77(6)*acd77(19)
      acd77(46)=-acd77(4)*acd77(16)
      acd77(45)=acd77(46)+acd77(40)+acd77(45)
      acd77(45)=acd77(39)*acd77(45)
      acd77(46)=acd77(9)*acd77(10)
      acd77(47)=acd77(11)*acd77(13)
      acd77(48)=-acd77(15)*acd77(19)
      acd77(48)=acd77(20)+acd77(48)
      acd77(48)=acd77(18)*acd77(48)
      acd77(49)=-acd77(15)*acd77(16)
      acd77(49)=acd77(17)+acd77(49)
      acd77(49)=acd77(14)*acd77(49)
      brack=acd77(41)+acd77(42)+acd77(43)+acd77(44)+acd77(45)-2.0_ki*acd77(46)+&
      &acd77(47)+acd77(48)+acd77(49)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd77h2
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(41) :: acd77
      complex(ki) :: brack
      acd77(1)=d(iv1,iv2)
      acd77(2)=abb77(11)
      acd77(3)=k2(iv1)
      acd77(4)=spval3k1(iv2)
      acd77(5)=abb77(38)
      acd77(6)=spval4k1(iv2)
      acd77(7)=abb77(40)
      acd77(8)=spval5k1(iv2)
      acd77(9)=abb77(26)
      acd77(10)=k2(iv2)
      acd77(11)=spval3k1(iv1)
      acd77(12)=spval4k1(iv1)
      acd77(13)=spval5k1(iv1)
      acd77(14)=spvak2l3(iv2)
      acd77(15)=abb77(16)
      acd77(16)=spvak2l3(iv1)
      acd77(17)=abb77(44)
      acd77(18)=spvak2k1(iv1)
      acd77(19)=spval3k2(iv2)
      acd77(20)=abb77(12)
      acd77(21)=spval4k2(iv2)
      acd77(22)=abb77(21)
      acd77(23)=spval4l3(iv2)
      acd77(24)=abb77(14)
      acd77(25)=spval5k2(iv2)
      acd77(26)=abb77(20)
      acd77(27)=spval5l3(iv2)
      acd77(28)=abb77(13)
      acd77(29)=spvak2k1(iv2)
      acd77(30)=spval3k2(iv1)
      acd77(31)=spval4k2(iv1)
      acd77(32)=spval4l3(iv1)
      acd77(33)=spval5k2(iv1)
      acd77(34)=spval5l3(iv1)
      acd77(35)=acd77(19)*acd77(20)
      acd77(36)=acd77(21)*acd77(22)
      acd77(37)=acd77(23)*acd77(24)
      acd77(38)=acd77(25)*acd77(26)
      acd77(39)=acd77(27)*acd77(28)
      acd77(35)=acd77(39)+acd77(38)+acd77(37)+acd77(36)+acd77(35)
      acd77(35)=acd77(18)*acd77(35)
      acd77(36)=acd77(30)*acd77(20)
      acd77(37)=acd77(31)*acd77(22)
      acd77(38)=acd77(32)*acd77(24)
      acd77(39)=acd77(33)*acd77(26)
      acd77(40)=acd77(34)*acd77(28)
      acd77(36)=acd77(40)+acd77(39)+acd77(38)+acd77(37)+acd77(36)
      acd77(36)=acd77(29)*acd77(36)
      acd77(37)=acd77(7)*acd77(6)
      acd77(38)=acd77(9)*acd77(8)
      acd77(39)=acd77(4)*acd77(5)
      acd77(37)=acd77(39)+acd77(37)+acd77(38)
      acd77(37)=acd77(3)*acd77(37)
      acd77(38)=acd77(12)*acd77(7)
      acd77(39)=acd77(13)*acd77(9)
      acd77(40)=acd77(11)*acd77(5)
      acd77(38)=acd77(40)+acd77(39)+acd77(38)
      acd77(38)=acd77(10)*acd77(38)
      acd77(39)=acd77(14)*acd77(12)
      acd77(40)=acd77(16)*acd77(6)
      acd77(39)=acd77(40)+acd77(39)
      acd77(39)=acd77(15)*acd77(39)
      acd77(40)=acd77(14)*acd77(13)
      acd77(41)=acd77(16)*acd77(8)
      acd77(40)=acd77(40)+acd77(41)
      acd77(40)=acd77(17)*acd77(40)
      acd77(41)=acd77(2)*acd77(1)
      brack=acd77(35)+acd77(36)+acd77(37)+acd77(38)+acd77(39)+acd77(40)+2.0_ki*&
      &acd77(41)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd77h2
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
      qshift = k2
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
end module     p0_ubaru_httbar_d77h2l1d
