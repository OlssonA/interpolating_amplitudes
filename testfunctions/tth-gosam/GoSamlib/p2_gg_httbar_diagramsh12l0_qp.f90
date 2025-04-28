module     p2_gg_httbar_diagramsh12l0_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12diagramsl0_qp.f90
   ! generator: buildfortranborn.py
   use p2_gg_httbar_color_qp, only: numcs
   use p2_gg_httbar_config, only: ki => ki_qp
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   complex(ki), dimension(numcs), parameter :: zero_col = 0.0_ki
   public :: amplitude
contains
!---#[ function amplitude:
   function amplitude()
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_config, only: debug_lo_diagrams, &
        & use_sorted_sum
      use p2_gg_httbar_accu_qp, only: sorted_sum
      use p2_gg_httbar_util_qp, only: inspect_lo_diagram
      implicit none
      complex(ki), dimension(numcs) :: amplitude
      complex(ki), dimension(54) :: abb
!      complex(ki), dimension(2,numcs) :: diagrams
      integer :: i
      amplitude(:) = 0.0_ki
      abb(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb(2)=es12**(-1)
      abb(3)=spak2l4**(-1)
      abb(4)=spak2l3**(-1)
      abb(5)=spbl3k2**(-1)
      abb(6)=spak2l5**(-1)
      abb(7)=sqrt(mT**2)
      abb(8)=1.0_ki/(mH**2-es34+es51-es23)
      abb(9)=spbl4k2**(-1)
      abb(10)=1.0_ki/(-mT**2+es34)
      abb(11)=1.0_ki/(-mT**2+es51)
      abb(12)=spbl5k2**(-1)
      abb(13)=1.0_ki/(es34-es51-es12)
      abb(14)=1.0_ki/(mT**2-es51+es23-es45)
      abb(15)=1.0_ki/2.0_ki*spak1k2
      abb(16)=spae1e2*abb(2)
      abb(15)=abb(15)*abb(16)
      abb(17)=-spbl4k1*abb(15)
      abb(18)=spbl4l3*spae2l3
      abb(19)=spae1k2*abb(11)
      abb(20)=-abb(19)*abb(18)
      abb(17)=abb(17)+abb(20)
      abb(17)=spbe2e1*abb(17)
      abb(20)=spbk1e1*spak1e2
      abb(21)=spbl5e1*spae2l5
      abb(22)=abb(20)+abb(21)
      abb(23)=abb(19)*spbl4e2
      abb(24)=-abb(22)*abb(23)
      abb(17)=abb(24)+abb(17)
      abb(17)=abb(10)*abb(17)
      abb(24)=spbl4e2*abb(8)
      abb(25)=spbl4e1*spae2l4
      abb(26)=abb(25)*abb(24)
      abb(27)=abb(26)*spae1k2
      abb(28)=abb(16)*spbe2e1*spak1k2
      abb(29)=1.0_ki/2.0_ki*abb(28)
      abb(30)=abb(29)*spbl4k1
      abb(31)=abb(27)-abb(30)
      abb(32)=spak2l3*spbl3e1
      abb(33)=abb(32)*abb(24)
      abb(34)=spae1e2*abb(33)
      abb(34)=abb(34)+abb(31)
      abb(34)=abb(1)*abb(34)
      abb(35)=abb(25)-abb(22)
      abb(23)=abb(23)*abb(8)
      abb(35)=abb(35)*abb(23)
      abb(36)=abb(10)*abb(11)
      abb(37)=spbe2e1*spae1k2
      abb(38)=abb(36)*abb(37)
      abb(39)=mH**2*abb(4)*abb(5)
      abb(40)=abb(39)*spbl4k2
      abb(41)=-spae2k2*abb(40)*abb(38)
      abb(17)=abb(41)+abb(34)+abb(35)+abb(17)
      abb(17)=abb(6)*abb(17)
      abb(34)=spae1l5*spbl5e2
      abb(35)=spae1k1*spbe2k1
      abb(34)=abb(34)+abb(35)
      abb(34)=spbl5e1*abb(34)
      abb(41)=abb(34)*abb(36)
      abb(42)=-abb(8)*abb(34)
      abb(43)=abb(24)*spbl5e1
      abb(44)=abb(43)*spae1l4
      abb(42)=abb(44)+abb(42)
      abb(42)=abb(11)*abb(42)
      abb(45)=spbl5l3*spae1l3
      abb(39)=abb(39)*spbl5k2
      abb(46)=-spae1k2*abb(39)
      abb(46)=-abb(45)+abb(46)
      abb(46)=spbe2e1*abb(8)*abb(46)
      abb(44)=abb(44)+abb(46)
      abb(44)=abb(1)*abb(44)
      abb(42)=abb(44)+abb(42)-abb(41)
      abb(42)=spae2k2*abb(42)
      abb(44)=spae1e2*spbl5e1
      abb(46)=spak2l3*spbl3e2
      abb(47)=abb(11)*abb(46)*abb(44)
      abb(48)=abb(29)*spbl5k1
      abb(47)=abb(47)-abb(48)
      abb(47)=abb(10)*abb(47)
      abb(49)=-abb(1)*abb(48)
      abb(42)=abb(42)+abb(47)+abb(49)
      abb(42)=abb(3)*abb(42)
      abb(36)=spbl4e2*abb(44)*abb(36)
      abb(44)=abb(43)*abb(1)
      abb(47)=spae1e2*abb(44)
      abb(17)=abb(42)+abb(17)+abb(36)+abb(47)
      abb(17)=abb(7)*abb(17)
      abb(42)=spae1l5*spbl5k2
      abb(47)=spae1k1*spbk2k1
      abb(42)=abb(42)+abb(47)
      abb(49)=abb(43)*abb(11)
      abb(42)=abb(42)*abb(49)
      abb(50)=abb(9)*abb(42)
      abb(51)=abb(45)*spbk2e1
      abb(52)=abb(9)*abb(51)
      abb(53)=spbl5e1*spae1l4
      abb(52)=abb(52)+abb(53)
      abb(24)=abb(24)*abb(1)
      abb(52)=abb(52)*abb(24)
      abb(50)=abb(52)+abb(50)-abb(41)
      abb(50)=spae2k2*abb(50)
      abb(52)=-abb(19)*abb(46)*abb(22)
      abb(29)=spbl3k1*abb(29)*spak2l3
      abb(52)=abb(52)-abb(29)
      abb(52)=abb(10)*abb(52)
      abb(53)=-spbk1e1*spak1l4
      abb(54)=-spbl5e1*spal4l5
      abb(53)=abb(53)+abb(54)
      abb(23)=abb(53)*abb(23)
      abb(33)=abb(1)*spae1l4*abb(33)
      abb(23)=abb(23)+abb(33)
      abb(23)=spae2k2*abb(23)
      abb(33)=-abb(1)*abb(29)
      abb(23)=abb(23)+abb(52)+abb(33)
      abb(23)=abb(6)*abb(23)
      abb(33)=abb(1)+abb(10)
      abb(28)=abb(33)*abb(28)
      abb(33)=-spbl5k1*abb(28)
      abb(23)=abb(23)+1.0_ki/2.0_ki*abb(33)+abb(50)
      abb(23)=abb(3)*abb(23)
      abb(33)=-spae2l4*spbl4k2*abb(12)*abb(19)*abb(43)
      abb(22)=-spbl4e2*abb(22)
      abb(43)=-spbl5e1*abb(12)*spbk2e2*abb(18)
      abb(22)=abb(43)+abb(22)
      abb(19)=abb(22)*abb(19)
      abb(19)=abb(19)-abb(30)
      abb(19)=abb(10)*abb(19)
      abb(22)=abb(1)*abb(31)
      abb(19)=abb(22)+abb(33)+abb(19)
      abb(19)=abb(6)*abb(19)
      abb(22)=spae2k2*abb(3)*abb(6)
      abb(31)=abb(7)**2
      abb(33)=abb(22)*abb(31)
      abb(43)=abb(37)*abb(8)
      abb(50)=abb(43)*abb(1)
      abb(38)=abb(50)+abb(38)
      abb(43)=-abb(11)*abb(43)
      abb(43)=abb(43)-abb(38)
      abb(43)=abb(43)*abb(33)
      abb(22)=abb(7)*abb(22)*mT
      abb(38)=-abb(38)*abb(22)
      abb(19)=abb(38)+abb(43)+abb(19)+abb(23)
      abb(19)=mT*abb(19)
      abb(17)=abb(17)+abb(19)
      abb(17)=mT*abb(17)
      abb(19)=spbl5l3*spak1l3
      abb(23)=abb(39)*spak1k2
      abb(19)=abb(19)+abb(23)
      abb(19)=abb(19)*spbl4k1
      abb(23)=spbl4k2*spak2l3*spbl5l3
      abb(19)=abb(19)-abb(23)
      abb(23)=1.0_ki/2.0_ki*spbe2e1
      abb(16)=abb(23)*abb(16)
      abb(19)=abb(19)*abb(16)
      abb(23)=abb(26)*abb(45)
      abb(26)=abb(39)*abb(27)
      abb(23)=-abb(19)+abb(23)+abb(26)
      abb(23)=abb(1)*abb(23)
      abb(26)=-abb(40)*abb(41)
      abb(24)=-abb(51)*abb(24)
      abb(24)=abb(24)-abb(42)+abb(26)
      abb(24)=spae2k2*abb(24)
      abb(26)=spbl4l3*spak1l3
      abb(27)=abb(40)*spak1k2
      abb(26)=abb(26)+abb(27)
      abb(26)=abb(26)*spbl5k1
      abb(27)=spbl5k2*spak2l3*spbl4l3
      abb(26)=abb(26)-abb(27)
      abb(16)=abb(26)*abb(16)
      abb(18)=-abb(11)*abb(18)*abb(34)
      abb(18)=abb(18)-abb(16)
      abb(18)=abb(10)*abb(18)
      abb(26)=abb(44)+abb(49)
      abb(26)=spae1e2*abb(26)
      abb(26)=abb(36)+abb(26)
      abb(26)=abb(26)*abb(31)
      abb(27)=spae1l5*spbl5l4
      abb(34)=spae1k1*spbl4k1
      abb(27)=abb(27)+abb(34)
      abb(27)=abb(49)*spae2l4*abb(27)
      abb(17)=abb(17)+abb(26)+abb(24)+abb(23)+abb(27)+abb(18)
      abb(18)=i_*e*gHT*gs**2
      abb(17)=abb(17)*abb(18)
      abb(23)=spbl5e2*abb(13)
      abb(24)=abb(21)*abb(23)
      abb(26)=abb(24)*spae1k2
      abb(27)=abb(26)-abb(48)
      abb(32)=abb(32)*abb(23)
      abb(34)=-spae1e2*abb(32)
      abb(34)=abb(34)-abb(27)
      abb(34)=abb(10)*abb(34)
      abb(15)=spbl5k1*abb(15)
      abb(36)=spbl5l3*spae2l3
      abb(38)=spae1k2*abb(14)
      abb(41)=abb(38)*abb(36)
      abb(15)=abb(15)+abb(41)
      abb(15)=spbe2e1*abb(15)
      abb(20)=abb(20)+abb(25)
      abb(25)=abb(38)*spbl5e2
      abb(41)=abb(20)*abb(25)
      abb(15)=abb(41)+abb(15)
      abb(15)=abb(1)*abb(15)
      abb(21)=-abb(21)+abb(20)
      abb(25)=abb(25)*abb(13)
      abb(21)=abb(21)*abb(25)
      abb(41)=abb(1)*abb(14)
      abb(42)=abb(41)*abb(37)
      abb(43)=spae2k2*abb(39)*abb(42)
      abb(15)=abb(43)+abb(15)+abb(21)+abb(34)
      abb(15)=abb(3)*abb(15)
      abb(21)=spae1l4*spbl4e2
      abb(21)=abb(21)+abb(35)
      abb(21)=spbl4e1*abb(21)
      abb(34)=abb(21)*abb(41)
      abb(35)=abb(13)*abb(21)
      abb(43)=abb(23)*spbl4e1
      abb(44)=abb(43)*spae1l5
      abb(35)=-abb(44)+abb(35)
      abb(35)=abb(14)*abb(35)
      abb(45)=spbl4l3*spae1l3
      abb(49)=spae1k2*abb(40)
      abb(49)=abb(45)+abb(49)
      abb(49)=spbe2e1*abb(13)*abb(49)
      abb(44)=-abb(44)+abb(49)
      abb(44)=abb(10)*abb(44)
      abb(35)=abb(34)+abb(35)+abb(44)
      abb(35)=spae2k2*abb(35)
      abb(44)=-spae1e2*abb(14)*spbl4e1*abb(46)
      abb(44)=abb(44)+abb(30)
      abb(44)=abb(1)*abb(44)
      abb(30)=abb(10)*abb(30)
      abb(30)=abb(35)+abb(30)+abb(44)
      abb(30)=abb(6)*abb(30)
      abb(35)=spbl4e1*abb(41)*spbl5e2
      abb(41)=abb(43)*abb(10)
      abb(35)=abb(35)+abb(41)
      abb(41)=-spae1e2*abb(35)
      abb(15)=abb(15)+abb(30)+abb(41)
      abb(15)=abb(7)*abb(15)
      abb(30)=abb(38)*abb(46)*abb(20)
      abb(30)=abb(30)+abb(29)
      abb(30)=abb(1)*abb(30)
      abb(41)=spbk1e1*spak1l5
      abb(44)=-spbl4e1*spal4l5
      abb(41)=abb(41)+abb(44)
      abb(25)=abb(41)*abb(25)
      abb(32)=-abb(10)*spae1l5*abb(32)
      abb(25)=abb(25)+abb(32)
      abb(25)=spae2k2*abb(25)
      abb(29)=abb(10)*abb(29)
      abb(25)=abb(25)+abb(29)+abb(30)
      abb(25)=abb(6)*abb(25)
      abb(29)=spae2l5*spbl5k2*abb(9)*abb(38)*abb(43)
      abb(27)=-abb(10)*abb(27)
      abb(20)=spbl5e2*abb(20)
      abb(30)=spbl4e1*abb(9)*spbk2e2*abb(36)
      abb(20)=abb(30)+abb(20)
      abb(20)=abb(20)*abb(38)
      abb(20)=abb(20)+abb(48)
      abb(20)=abb(1)*abb(20)
      abb(20)=abb(25)+abb(20)+abb(29)+abb(27)
      abb(20)=abb(3)*abb(20)
      abb(25)=spae1l4*spbl4k2
      abb(25)=abb(25)+abb(47)
      abb(27)=abb(43)*abb(14)
      abb(25)=abb(25)*abb(27)
      abb(29)=-abb(12)*abb(25)
      abb(30)=abb(45)*spbk2e1
      abb(32)=-abb(12)*abb(30)
      abb(38)=-spbl4e1*spae1l5
      abb(32)=abb(32)+abb(38)
      abb(23)=abb(23)*abb(10)
      abb(32)=abb(32)*abb(23)
      abb(29)=abb(34)+abb(29)+abb(32)
      abb(29)=spae2k2*abb(29)
      abb(28)=spbl4k1*abb(28)
      abb(28)=1.0_ki/2.0_ki*abb(28)+abb(29)
      abb(28)=abb(6)*abb(28)
      abb(29)=abb(37)*abb(13)
      abb(32)=abb(29)*abb(10)
      abb(32)=abb(32)+abb(42)
      abb(29)=abb(14)*abb(29)
      abb(29)=abb(29)+abb(32)
      abb(29)=abb(29)*abb(33)
      abb(22)=abb(32)*abb(22)
      abb(20)=abb(22)+abb(29)+abb(28)+abb(20)
      abb(20)=mT*abb(20)
      abb(15)=abb(15)+abb(20)
      abb(15)=mT*abb(15)
      abb(20)=-abb(24)*abb(45)
      abb(22)=-abb(40)*abb(26)
      abb(16)=abb(16)+abb(20)+abb(22)
      abb(16)=abb(10)*abb(16)
      abb(20)=abb(30)*abb(23)
      abb(22)=abb(39)*abb(34)
      abb(20)=abb(22)+abb(25)+abb(20)
      abb(20)=spae2k2*abb(20)
      abb(21)=abb(14)*abb(36)*abb(21)
      abb(19)=abb(21)+abb(19)
      abb(19)=abb(1)*abb(19)
      abb(21)=spae1l4*spbl5l4
      abb(22)=-spae1k1*spbl5k1
      abb(21)=abb(21)+abb(22)
      abb(21)=abb(27)*spae2l5*abb(21)
      abb(22)=-abb(27)-abb(35)
      abb(22)=abb(31)*spae1e2*abb(22)
      abb(15)=abb(15)+abb(22)+abb(20)+abb(19)+abb(21)+abb(16)
      abb(15)=abb(15)*abb(18)
      amplitude=c2*abb(17)+c1*abb(15)
      if (debug_lo_diagrams) then
         write(*,*) "Using Born optimization, debug_lo_diagrams not implemented&
         &."
      end if
!      if (use_sorted_sum) then
!         do i=1,numcs
!            amplitude(i) = sorted_sum(diagrams(i))
!         end do
!      else
!         do i=1,numcs
!            amplitude(i) = sum(diagrams(i))
!         end do
!      end if
   end function     amplitude
!---#] function amplitude:
end module p2_gg_httbar_diagramsh12l0_qp
